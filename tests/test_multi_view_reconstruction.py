import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from shape_tracking.geometry import cumulative_arclength
from shape_tracking.multi_view_reconstruction import (
    ViewObservation,
    _soft_l1_residual,
    fit_multi_view_spline,
    projected_route_topology_weight,
    triangulate_material_curve,
)
from shape_tracking.multi_image_sequence import (
    MultiCameraConfig,
    _associate_cross_camera_marker_identities,
    _fit_reference_from_available_rigs,
    _fit_ordered_disparity_spline,
    _ordered_stereo_curve_candidate,
    _pair_map,
    _repair_cross_camera_marker_centers,
    _robust_rigid_alignment,
    _result_requires_reacquisition,
    _result_rig_errors,
    _select_consistent_rig,
    _select_ordered_stereo_curve,
)
from shape_tracking.session import project_points


def _look_at(center_mm, target_mm):
    center = np.asarray(center_mm, float) * 1e-3
    target = np.asarray(target_mm, float) * 1e-3
    z = target - center
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.vstack([x, y, z])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ center
    return transform


class MultiViewReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0],
            [0.0, 0.0, 1.0],
        ])
        u = np.linspace(0.0, 1.0, 64)
        raw = np.column_stack([
            18.0 * np.sin(np.pi * u),
            5.0 * np.sin(2.0 * np.pi * u),
            10.0 + 50.0 * u,
        ])
        # Normalize the synthetic material length to the marked 57 mm segment.
        raw[:, :2] *= 57.0 / cumulative_arclength(raw)[-1]
        self.points = raw
        centers = [
            (-160, -170, 20), (-50, -220, 20),
            (50, -220, 20), (160, -170, 20),
        ]
        self.observations = []
        for index, center in enumerate(centers):
            transform = _look_at(center, (0, 20, 35))
            pixels, in_front = project_points(self.K, transform, self.points)
            self.assertTrue(np.all(in_front))
            self.observations.append(ViewObservation(
                f"view_{index}", self.K, transform, pixels))

    def test_fit_reference_is_joint_until_only_one_rig_is_usable(self):
        reference, rigs = _fit_reference_from_available_rigs(None, 3)
        self.assertIsNone(reference)
        self.assertEqual(rigs, {"primary", "oblique"})

        reference, rigs = _fit_reference_from_available_rigs(None, 2)
        self.assertEqual(reference, "oblique")
        self.assertEqual(rigs, {"oblique"})

        reference, rigs = _fit_reference_from_available_rigs("primary", 3)
        self.assertEqual(reference, "primary")
        self.assertEqual(rigs, {"primary"})

    def test_robust_rigid_alignment_recovers_small_extrinsic_error(self):
        rng = np.random.default_rng(4)
        source = rng.normal(size=(500, 3)) * np.array([20.0, 30.0, 15.0])
        angle = np.deg2rad(0.8)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        translation = np.array([-0.8, 1.9, 1.1])
        target = (rotation @ source.T).T + translation
        target += rng.normal(scale=0.05, size=target.shape)
        target[::23] += rng.normal(scale=20.0, size=target[::23].shape)
        fitted_rotation, fitted_translation, inliers = (
            _robust_rigid_alignment(source, target))
        self.assertGreater(int(np.count_nonzero(inliers)), 450)
        np.testing.assert_allclose(fitted_rotation, rotation, atol=5e-4)
        np.testing.assert_allclose(
            fitted_translation, translation, atol=0.03)

    def test_material_order_dlt_is_a_finite_initializer(self):
        reconstructed = triangulate_material_curve(self.observations)
        self.assertEqual(reconstructed.shape, (64, 3))
        error = np.sqrt(np.mean(np.sum(
            (reconstructed - self.points) ** 2, axis=1)))
        # Each image route is resampled by projected arclength, so normalized
        # indices are only an approximate material correspondence. The DLT is
        # an initializer, not the final reconstruction.
        self.assertLess(error, 2.0)

    def test_common_spline_recovers_curve_with_one_downweighted_bad_view(self):
        rng = np.random.default_rng(4)
        observations = []
        for index, observation in enumerate(self.observations):
            noisy = observation.centerline_xy + rng.normal(
                scale=0.15, size=observation.centerline_xy.shape)
            if index == 3:
                noisy = noisy + np.array([25.0, -20.0])
            observations.append(ViewObservation(
                observation.view_id, observation.K,
                observation.camera_T_base, noisy,
                weight=0.05 if index == 3 else 1.0))
        initial = self.points + rng.normal(scale=1.0, size=self.points.shape)
        result = fit_multi_view_spline(
            initial, observations,
            nominal_length_mm=cumulative_arclength(self.points)[-1],
            max_nfev=45)
        self.assertTrue(result.optimizer_success)
        error = np.sqrt(np.mean(np.sum(
            (result.points_base_mm - self.points) ** 2, axis=1)))
        self.assertLess(error, 2.0)
        self.assertLess(result.final_symmetric_mean_px, 10.0)

    def test_gated_quadratic_stereo_does_not_unrobustify_image_routes(self):
        # Reproduce the important trust asymmetry: ordered stereo depth has
        # passed its frame gates, while an image route may still be the wrong
        # branch and must retain a saturating residual.
        observations = []
        for index, observation in enumerate(self.observations[:2]):
            shifted = observation.centerline_xy.copy()
            if index:
                shifted += np.array([20.0, -20.0])
            observations.append(ViewObservation(
                observation.view_id, observation.K,
                observation.camera_T_base, shifted,
                weight=0.002 if index else 0.10))
        initial = self.points + np.array([3.0, -2.0, 4.0])
        nominal = cumulative_arclength(self.points)[-1]
        robust_depth = fit_multi_view_spline(
            initial, observations, nominal_length_mm=nominal,
            stereo_curve_points_base_mm=self.points,
            stereo_curve_weight=2.0, stereo_curve_sigma_mm=1.0,
            stereo_curve_quadratic=False, max_nfev=40)
        trusted_depth = fit_multi_view_spline(
            initial, observations, nominal_length_mm=nominal,
            stereo_curve_points_base_mm=self.points,
            stereo_curve_weight=2.0, stereo_curve_sigma_mm=1.0,
            stereo_curve_quadratic=True, max_nfev=40)
        robust_error = np.sqrt(np.mean(np.sum(
            (robust_depth.points_base_mm - self.points) ** 2, axis=1)))
        trusted_error = np.sqrt(np.mean(np.sum(
            (trusted_depth.points_base_mm - self.points) ** 2, axis=1)))
        self.assertTrue(trusted_depth.optimizer_success)
        self.assertLess(trusted_error, robust_error - 0.2)

    def test_axial_endpoints_prevent_a_low_mean_terminal_tail(self):
        marker_indices = [0, 12, 34, 63]
        observations = [
            ViewObservation(
                item.view_id, item.K, item.camera_T_base,
                item.centerline_xy,
                axial_markers_xy=item.centerline_xy[marker_indices])
            for item in self.observations]
        initial = self.points.copy()
        # A wrong depth tail can project near most of a route while its last
        # few samples miss the observed tip. Endpoint terms must remain
        # influential even when dense image residuals are robustified.
        initial[-12:, 2] += np.linspace(0.0, 22.0, 12)
        nominal = cumulative_arclength(self.points)[-1]
        result = fit_multi_view_spline(
            initial, observations, nominal_length_mm=nominal, max_nfev=60)
        self.assertTrue(result.optimizer_success)
        self.assertLess(max(result.view_terminal_end_px.values()), 6.0)

    def test_soft_l1_transform_jacobian_matches_finite_difference(self):
        values = np.array([-8.0, -0.4, 0.0, 0.7, 12.0])
        transformed, derivative = _soft_l1_residual(values)
        epsilon = 1e-6
        plus = _soft_l1_residual(values + epsilon)[0]
        minus = _soft_l1_residual(values - epsilon)[0]
        numerical = (plus - minus) / (2.0 * epsilon)
        np.testing.assert_allclose(derivative, numerical, atol=5e-5)
        self.assertTrue(np.all(np.isfinite(transformed)))

    def test_recorded_frame_pair_table_is_authoritative(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "camera_frame_pairs.csv").write_text(
                "reference_frame,reference_timestamp_ns,secondary_frame,"
                "secondary_timestamp_ns,timestamp_delta_ns\n"
                "4,1000,7,1012,12\n"
                "5,1033,,,\n",
                encoding="utf-8")
            self.assertEqual(_pair_map(root), {4: (7, 1000, 1012)})

    def test_tight_v_projection_is_softly_downweighted(self):
        smooth = np.column_stack([
            np.linspace(0.0, 100.0, 96),
            8.0 * np.sin(np.linspace(0.0, np.pi, 96)),
        ])
        arm = np.linspace(0.0, 1.0, 48)
        tight_v = np.vstack([
            np.column_stack([50.0 * arm, 50.0 * arm]),
            np.column_stack([50.0 * (1.0 - arm[1:]),
                             50.0 * arm[1:] + 2.0]),
        ])
        smooth_weight = projected_route_topology_weight(smooth)
        tight_v_weight = projected_route_topology_weight(tight_v)
        self.assertGreater(smooth_weight, 0.9)
        self.assertLess(tight_v_weight, 0.5)
        self.assertGreaterEqual(tight_v_weight, 0.08)

    def test_marker_order_softly_exposes_a_shortcut_route(self):
        route = np.column_stack([
            np.linspace(0.0, 100.0, 96), np.zeros(96)])
        ordered = np.array([
            [0.0, 2.0], [20.0, -2.0], [55.0, 2.0], [100.0, -2.0]])
        collapsed = ordered.copy()
        collapsed[1] = [65.0, 1.0]
        self.assertGreater(
            projected_route_topology_weight(route, ordered), 0.8)
        self.assertLess(
            projected_route_topology_weight(route, collapsed), 0.6)

    def test_cross_camera_marker_repair_rejects_one_wrong_eye(self):
        count = 7
        marker_indices = [0, 12, 34, 63]
        centers = np.empty((count, 4, 4, 2), dtype=float)
        for view, observation in enumerate(self.observations):
            marker_pixels = observation.centerline_xy[marker_indices]
            centers[:, view] = marker_pixels[None]
        expected = centers[3, 0, 2].copy()
        centers[3, 0, 2] += [55.0, -35.0]
        timestamps = np.arange(count, dtype=np.int64) * 33_333_333
        repaired, rejected = _repair_cross_camera_marker_centers(
            centers, self.observations, timestamps, MultiCameraConfig())
        self.assertTrue(rejected[3, 0, 2])
        np.testing.assert_allclose(repaired[3, 0, 2], expected, atol=1e-6)

    def test_rig_error_is_unweighted_and_exposes_failed_camera(self):
        result = SimpleNamespace(
            view_model_mean_px={
                "primary_left": 22.0, "primary_right": 18.0,
                "oblique_left": 1.0, "oblique_right": 2.0},
            view_coverage_mean_px={
                "primary_left": 20.0, "primary_right": 16.0,
                "oblique_left": 1.0, "oblique_right": 2.0})
        errors = _result_rig_errors(result)
        self.assertAlmostEqual(errors["primary"], 19.0)
        self.assertAlmostEqual(errors["oblique"], 1.5)

    def test_excluded_rig_does_not_force_permanent_reacquisition(self):
        result = SimpleNamespace(
            optimizer_success=True,
            points_base_mm=np.zeros((64, 3)),
            view_model_mean_px={
                "primary_left": 24.0, "primary_right": 20.0,
                "oblique_left": 1.0, "oblique_right": 1.2},
            view_coverage_mean_px={
                "primary_left": 18.0, "primary_right": 16.0,
                "oblique_left": 1.3, "oblique_right": 1.5},
            view_terminal_start_px={
                "primary_left": 25.0, "primary_right": 22.0,
                "oblique_left": 1.0, "oblique_right": 1.0},
            view_terminal_end_px={
                "primary_left": 30.0, "primary_right": 26.0,
                "oblique_left": 1.5, "oblique_right": 1.5},
            length_residual_mm=0.5)
        config = MultiCameraConfig()
        self.assertFalse(_result_requires_reacquisition(
            result, config, trusted_rigs={"oblique"}))
        self.assertTrue(_result_requires_reacquisition(
            result, config, trusted_rigs={"primary", "oblique"}))

    def test_dual_camera_temporal_blend_defaults_suppress_frame_jitter(self):
        config = MultiCameraConfig()
        self.assertEqual(config.temporal_observation_blend, 0.15)
        self.assertEqual(config.temporal_terminal_observation_blend, 0.15)
        self.assertEqual(config.temporal_terminal_cutoff_hz, 3.0)

    def test_ordered_stereo_depth_uses_whichever_rig_is_trusted(self):
        left_transform = np.eye(4)
        right_transform = np.eye(4)
        right_transform[0, 3] = -0.030
        u = np.linspace(0.0, 1.0, 64)
        points = np.column_stack([
            8.0 * np.sin(np.pi * u),
            18.0 * u,
            230.0 + 48.0 * u,
        ])
        primary = []
        oblique = []
        for rig, target in (("primary", primary), ("oblique", oblique)):
            for eye, transform in (
                    ("left", left_transform),
                    ("right", right_transform)):
                pixels, in_front = project_points(
                    self.K, transform, points)
                self.assertTrue(np.all(in_front))
                target.append(ViewObservation(
                    f"{rig}_{eye}", self.K, transform, pixels,
                    topology_weight=1.0))
        config = MultiCameraConfig()
        primary_candidate = _ordered_stereo_curve_candidate(
            primary, "primary", config)
        self.assertIsNotNone(primary_candidate)
        reconstructed = primary_candidate["points_base_mm"]
        distances = np.linalg.norm(
            reconstructed[:, None, :] - points[None, :, :], axis=2)
        error = 0.5 * (
            float(np.mean(np.min(distances, axis=1)))
            + float(np.mean(np.min(distances, axis=0))))
        self.assertLess(error, 0.5)

        # Names do not establish preference. The explicitly trusted complete
        # rig supplies depth, and the direction reverses symmetrically.
        observations = primary + oblique
        selected = _select_ordered_stereo_curve(
            observations, "primary", None, config)
        self.assertEqual(selected["rig"], "primary")
        selected = _select_ordered_stereo_curve(
            observations, "oblique", "primary", config)
        self.assertEqual(selected["rig"], "oblique")

        # Corruption is rig-local. A corrupt trusted rig is excluded and the
        # other independently valid camera pair becomes the reference.
        selected = _select_ordered_stereo_curve(
            observations, "primary", "primary", config,
            corrupted_rigs={"primary"})
        self.assertEqual(selected["rig"], "oblique")
        self.assertEqual(selected["available_rig_mask"], 2)
        selected = _select_ordered_stereo_curve(
            observations, "oblique", "oblique", config,
            corrupted_rigs={"oblique"})
        self.assertEqual(selected["rig"], "primary")
        self.assertEqual(selected["available_rig_mask"], 1)

        selected = _select_ordered_stereo_curve(
            observations, None, None, config,
            corrupted_rigs={"primary", "oblique"})
        self.assertIsNone(selected)

        ill_primary = [replace(item, topology_weight=0.08)
                       for item in primary]
        selected = _select_ordered_stereo_curve(
            ill_primary + oblique, None, "primary", config)
        self.assertEqual(selected["rig"], "oblique")

    def test_temporal_disparity_spline_rejects_pointwise_depth_spikes(self):
        config = replace(
            MultiCameraConfig(), ordered_disparity_temporal_weight=1.5)
        count = config.ordered_stereo_samples
        u = np.linspace(0.0, 1.0, count)
        left = np.column_stack([400.0 + 20.0 * u, 200.0 + 80.0 * u])
        true_disparity = 32.0 + 3.0 * u + 0.5 * np.sin(np.pi * u)
        right = left.copy()
        right[:, 0] -= true_disparity
        indices = np.arange(count)
        clean = _fit_ordered_disparity_spline(
            left, right, indices, count, config)
        corrupted = right.copy()
        corrupted[12::9, 0] -= 15.0
        robust = _fit_ordered_disparity_spline(
            left, corrupted, indices, count, config,
            previous_coefficients=clean["coefficients_px"])
        fitted = left[:, 0] - robust["disparity_px"]
        error = np.abs(fitted - right[:, 0])
        self.assertLess(float(np.percentile(error, 95)), 1.0)
        self.assertTrue(robust["temporal_used"])

    def test_default_disparity_fit_does_not_lag_toward_stale_state(self):
        config = MultiCameraConfig()
        count = config.ordered_stereo_samples
        u = np.linspace(0.0, 1.0, count)
        left = np.column_stack([400.0 + 20.0 * u, 200.0 + 80.0 * u])
        disparity = 36.0 + 2.0 * u
        right = left.copy()
        right[:, 0] -= disparity
        stale = np.full(config.ordered_disparity_basis_count, 20.0)
        result = _fit_ordered_disparity_spline(
            left, right, np.arange(count), count, config,
            previous_coefficients=stale)
        self.assertFalse(result["temporal_used"])
        self.assertLess(float(np.median(np.abs(
            result["disparity_px"] - disparity))), 0.25)

    def test_cross_rig_tip_disagreement_selects_temporally_consistent_rig(self):
        good_tip = self.points[-1]
        bad_tip = good_tip + np.array([20.0, 0.0, 0.0])
        observations = []
        names = ("primary_left", "primary_right",
                 "oblique_left", "oblique_right")
        for index, (name, item) in enumerate(zip(names, self.observations)):
            markers = np.full((4, 2), np.nan)
            tip = good_tip if index < 2 else bad_tip
            markers[3] = project_points(
                item.K, item.camera_T_base, tip[None])[0][0]
            observations.append(ViewObservation(
                name, item.K, item.camera_T_base,
                item.centerline_xy, axial_markers_xy=markers))
        selected, endpoints, disagreement = _select_consistent_rig(
            observations, good_tip, maximum_disagreement_mm=8.0)
        self.assertEqual(selected, "primary")
        self.assertGreater(disagreement, 15.0)
        np.testing.assert_allclose(endpoints["primary"], good_tip, atol=1e-6)
        selected, _, _ = _select_consistent_rig(
            observations, bad_tip, maximum_disagreement_mm=8.0)
        self.assertEqual(selected, "oblique")

    def test_complete_rig_repairs_shifted_ids_without_inventing_tip_component(self):
        names = ("primary_left", "primary_right",
                 "oblique_left", "oblique_right")
        observations = [
            ViewObservation(name, item.K, item.camera_T_base,
                            item.centerline_xy)
            for name, item in zip(names, self.observations)]
        marker_points = self.points[[0, 12, 34, 63]]
        projected = np.stack([
            project_points(item.K, item.camera_T_base, marker_points)[0]
            for item in observations])
        count = 5
        centers = np.full((count, 4, 4, 2), np.nan)
        # The complete oblique rig has the correct ordered sequence.
        centers[:, 2:] = projected[2:][None]
        # Primary slot 1 duplicates marker 0 in only one eye; slots 2/3 are
        # actually physical markers 1/2, and physical marker 3 is absent.
        centers[:, 0, 0] = projected[0, 0]
        centers[:, 1, 0] = projected[1, 0]
        centers[:, 0, 1] = projected[0, 0] + [1.0, 0.5]
        centers[:, 0, 2] = projected[0, 1]
        centers[:, 1, 2] = projected[1, 1]
        centers[:, 0, 3] = projected[0, 2]
        centers[:, 1, 3] = projected[1, 2]
        widths = np.full((count, 4, 4), 8.0)
        widths[:, 2:, 3] = 24.0
        confidence = np.full((count, 4, 4), 0.8)
        timestamps = np.arange(count, dtype=np.int64) * 33_333_333
        (assigned, source, inferred, reference, _) = (
            _associate_cross_camera_marker_identities(
                centers, widths, confidence, observations, timestamps))
        self.assertTrue(np.all(reference == 2))
        np.testing.assert_array_equal(source[2, 0], [0, 2, 3, -1])
        np.testing.assert_array_equal(source[2, 1], [0, 2, 3, -1])
        self.assertTrue(np.all(inferred[2, :2, 3] == 1))
        np.testing.assert_allclose(
            assigned[2, 0, 3], projected[0, 3], atol=1e-6)
        np.testing.assert_allclose(
            assigned[2, 1, 3], projected[1, 3], atol=1e-6)

    def test_primary_can_be_temporary_marker_identity_reference(self):
        names = ("primary_left", "primary_right",
                 "oblique_left", "oblique_right")
        observations = [
            ViewObservation(name, item.K, item.camera_T_base,
                            item.centerline_xy)
            for name, item in zip(names, self.observations)]
        marker_points = self.points[[0, 12, 34, 63]]
        projected = np.stack([
            project_points(item.K, item.camera_T_base, marker_points)[0]
            for item in observations])
        count = 3
        centers = np.full((count, 4, 4, 2), np.nan)
        centers[:, :2] = projected[:2][None]
        # Mirror the failure in the oblique rig: slots 2/3 contain physical
        # rings 1/2 and the actual distal ring is absent.
        centers[:, 2, 0] = projected[2, 0]
        centers[:, 3, 0] = projected[3, 0]
        centers[:, 2, 1] = projected[2, 0] + [1.0, 0.5]
        centers[:, 2, 2] = projected[2, 1]
        centers[:, 3, 2] = projected[3, 1]
        centers[:, 2, 3] = projected[2, 2]
        centers[:, 3, 3] = projected[3, 2]
        widths = np.full((count, 4, 4), 8.0)
        confidence = np.full((count, 4, 4), 0.8)
        timestamps = np.arange(count, dtype=np.int64) * 33_333_333
        (assigned, source, inferred, reference, _) = (
            _associate_cross_camera_marker_identities(
                centers, widths, confidence, observations, timestamps))
        self.assertTrue(np.all(reference == 1))
        np.testing.assert_array_equal(source[1, 2], [0, 2, 3, -1])
        np.testing.assert_array_equal(source[1, 3], [0, 2, 3, -1])
        self.assertTrue(np.all(inferred[1, 2:, 3] == 1))
        np.testing.assert_allclose(
            assigned[1, 2, 3], projected[2, 3], atol=1e-6)
        np.testing.assert_allclose(
            assigned[1, 3, 3], projected[3, 3], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
