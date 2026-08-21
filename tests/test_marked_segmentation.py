import unittest
from dataclasses import replace

import cv2
import numpy as np

from shape_tracking.marked_segmentation import (
    catheter_color_likelihood,
    decode_marker_candidates,
    extract_marked_chromatic_result,
    refine_marked_stereo_pair,
)
from shape_tracking.sequence_reconstruction import (
    project_camera_points,
    reconstruct_disparity_anchored,
)
from shape_tracking.segmentation import Centerline


class MarkedSegmentationTests(unittest.TestCase):
    @staticmethod
    def _image():
        image = np.full((140, 410, 3), 225, dtype=np.uint8)
        cv2.line(image, (20, 80), (145, 80), (90, 20, 15), 9,
                 cv2.LINE_AA)
        cv2.line(image, (145, 80), (375, 55), (255, 165, 15), 9,
                 cv2.LINE_AA)
        # Interface, two thin body rings, and the wide distal probe marker.
        for x, thickness in ((145, 9), (185, 7), (260, 7), (355, 18)):
            y = int(round(80.0 - (x - 145) * 25.0 / 230.0))
            cv2.circle(image, (x, y), max(5, thickness // 2), (0, 0, 245), -1)
        cv2.circle(image, (378, 54), 6, (0, 220, 255), -1)
        return image

    def test_color_backend_extracts_complete_shaft_and_four_markers(self):
        result = extract_marked_chromatic_result(
            self._image(), (0, 0, 410, 140), np.array([18.0, 80.0]))
        self.assertGreater(len(result.material.points), 250)
        self.assertEqual(int(np.sum(result.marker_observed)), 4)
        self.assertTrue(np.all(np.diff(result.marker_centers_xy[:, 0]) > 0))
        self.assertLess(abs(
            result.material.points[result.material.distal_boundary_index, 0]
            - 145.0), 8.0)
        self.assertIsNone(result.yellow_tip_xy)
        self.assertLess(np.linalg.norm(
            result.material.points[-1]
            - result.marker_centers_xy[3]), 3.0)

    def test_distal_ring_and_yellow_tip_reject_long_background_branch(self):
        image = self._image()
        # Model the saturated antialiased edge that becomes connected when the
        # catheter crosses a ChArUco vertex.  It is longer than the real
        # ring-to-tip segment, so an unconstrained longest path follows it.
        cv2.line(image, (348, 58), (400, 125), (170, 105, 35), 5,
                 cv2.LINE_AA)
        result = extract_marked_chromatic_result(
            image, (0, 0, 410, 140), np.array([18.0, 80.0]))
        endpoint = result.material.points[-1]
        if result.yellow_tip_xy is not None:
            self.assertLess(np.linalg.norm(endpoint - result.yellow_tip_xy), 8.0)
        # Rejecting an ambiguous yellow observation and stopping on the wide
        # ring is safer than forcing the path onto the background branch.
        self.assertGreater(endpoint[0], 345.0)
        self.assertLess(endpoint[1], 75.0)

    def test_large_interface_ring_is_not_mistaken_for_distal_ring(self):
        image = self._image()
        # Foreshortening can make marker 0 occupy more pixels than marker 3.
        # Distal identity must still include position along the catheter.
        cv2.circle(image, (145, 80), 11, (0, 0, 245), -1)
        result = extract_marked_chromatic_result(
            image, (0, 0, 410, 140), np.array([18.0, 80.0]))
        self.assertTrue(result.marker_observed[0])
        self.assertTrue(result.marker_observed[3])
        self.assertLess(result.marker_centers_xy[0, 0], 180.0)
        self.assertGreater(result.marker_centers_xy[3, 0], 330.0)

    def test_off_axis_yellow_board_vertex_is_not_a_tip_anchor(self):
        image = self._image()
        # Remove the real exposed yellow dot, then add a strongly saturated
        # yellow patch close to the distal ring but perpendicular to the local
        # shaft direction, as occurs at a warm ChArUco vertex.
        cv2.circle(image, (378, 54), 8, (225, 225, 225), -1)
        cv2.circle(image, (355, 92), 5, (0, 210, 255), -1)
        cv2.line(image, (355, 92), (355, 72), (0, 210, 255), 3,
                 cv2.LINE_AA)
        result = extract_marked_chromatic_result(
            image, (0, 0, 410, 140), np.array([18.0, 80.0]))
        self.assertIsNone(result.yellow_tip_xy)
        # With no supported yellow observation, stay at the wide distal band;
        # never bend upward to the distractor.
        self.assertLess(result.material.points[-1, 1], 75.0)

    def test_static_yellow_board_vertex_is_removed_by_background(self):
        image = self._image()
        background = np.full_like(image, 225)
        # Remove the real dome and place the same warm board vertex in both the
        # current image and temporal-median background.
        cv2.circle(image, (378, 54), 8, (225, 225, 225), -1)
        for target in (image, background):
            cv2.circle(target, (365, 82), 5, (0, 210, 255), -1)
        result = extract_marked_chromatic_result(
            image, (0, 0, 410, 140), np.array([18.0, 80.0]),
            background_bgr=background, minimum_background_difference=18)
        self.assertIsNone(result.yellow_tip_xy)

    def test_static_chromatic_board_fringe_is_removed_by_background(self):
        image = self._image()
        background = np.full_like(image, 225)
        # Same saturated fringe is present in the fixed scene and current
        # frame, while the catheter exists only in the current frame.
        for target in (image, background):
            cv2.line(target, (330, 60), (400, 125), (170, 105, 35), 7,
                     cv2.LINE_AA)
        result = extract_marked_chromatic_result(
            image, (0, 0, 410, 140), np.array([18.0, 80.0]),
            background_bgr=background,
            minimum_background_difference=18)
        endpoint = result.material.points[-1]
        self.assertLess(abs(endpoint[0] - 355.0), 5.0)
        self.assertLess(endpoint[1], 75.0)
        self.assertLess(result.mask_area_px, 5000)

    def test_grey_board_edge_has_low_color_likelihood(self):
        image = np.full((80, 160, 3), 225, dtype=np.uint8)
        cv2.rectangle(image, (15, 10), (70, 70), (25, 25, 25), -1)
        # Mild channel imbalance represents demosaicing at a high-contrast
        # board edge; the actual catheter has much stronger directed chroma.
        image[:, 72:75] = (80, 66, 58)
        cv2.line(image, (90, 50), (145, 35), (255, 165, 15), 8,
                 cv2.LINE_AA)
        likelihood = catheter_color_likelihood(image)
        self.assertLess(float(np.max(likelihood[:, 72:75])), 0.08)
        self.assertGreater(float(np.median(likelihood[35:51, 95:140])), 0.25)

    def test_dark_color_cast_board_pixel_is_not_paint(self):
        image = np.full((40, 80, 3), (35, 18, 17), dtype=np.uint8)
        image[:, 45:] = (90, 20, 15)
        likelihood = catheter_color_likelihood(image)
        self.assertLess(float(np.max(likelihood[:, :40])), 0.05)
        self.assertGreater(float(np.median(likelihood[:, 50:])), 0.35)

    def test_continuous_route_bridges_a_short_neutral_gap(self):
        image = self._image()
        # A specular/neutral patch severs every binary color component, but the
        # continuous-cost route should cross the short gap and retain the
        # marker-centered distal endpoint.
        cv2.rectangle(image, (210, 70), (218, 86), (225, 225, 225), -1)
        result = extract_marked_chromatic_result(
            image, (0, 0, 410, 140), np.array([18.0, 80.0]))
        self.assertGreater(len(result.material.points), 250)
        self.assertLess(np.linalg.norm(
            result.material.points[-1] - result.marker_centers_xy[3]), 3.0)

    def test_cross_eye_marker_recovery_does_not_change_segmentation(self):
        left_image = self._image()
        right_image = self._image()
        left = extract_marked_chromatic_result(
            left_image, (0, 0, 410, 140), np.array([18.0, 80.0]))
        right = extract_marked_chromatic_result(
            right_image, (0, 0, 410, 140), np.array([18.0, 80.0]))
        left_mask = left.material.mask.copy()
        right_mask = right.material.mask.copy()
        left_points = left.material.points.copy()
        right_points = right.material.points.copy()

        refined_left, refined_right = refine_marked_stereo_pair(
            left_image, right_image, left, right,
            (0, 0, 410, 140), (0, 0, 410, 140),
            base_disparity_px=0.0)

        np.testing.assert_array_equal(refined_left.material.mask, left_mask)
        np.testing.assert_array_equal(refined_right.material.mask, right_mask)
        np.testing.assert_allclose(refined_left.material.points, left_points)
        np.testing.assert_allclose(refined_right.material.points, right_points)

    def test_cross_eye_recovery_preserves_confident_local_endpoint_markers(self):
        image = self._image()
        left = extract_marked_chromatic_result(
            image, (0, 0, 410, 140), np.array([18.0, 80.0]))
        right = extract_marked_chromatic_result(
            image, (0, 0, 410, 140), np.array([18.0, 80.0]))
        # Make the right curve the reference view, so the left observations
        # are the target that cross-eye recovery would previously overwrite.
        right_points = right.material.points.copy()
        right_points[:, 0] *= 1.02
        right = replace(
            right,
            material=replace(
                right.material,
                centerline=Centerline(
                    right_points, right.material.centerline.roi,
                    right.material.centerline.mask,
                    right.material.centerline.radius_px)))
        local_centers = left.marker_centers_xy.copy()
        local_centers[0] += [0.7, 1.4]
        local_centers[3] += [-0.8, 1.2]
        local_confidence = left.marker_confidence.copy()
        local_confidence[[0, 3]] = [0.65, 0.95]
        left = replace(
            left, marker_centers_xy=local_centers,
            marker_confidence=local_confidence)

        refined_left, _ = refine_marked_stereo_pair(
            image, image, left, right,
            (0, 0, 410, 140), (0, 0, 410, 140),
            base_disparity_px=0.0)

        np.testing.assert_allclose(
            refined_left.marker_centers_xy[[0, 3]],
            local_centers[[0, 3]])
        np.testing.assert_allclose(
            refined_left.marker_confidence[[0, 3]], [0.65, 0.95])

    def test_distal_marker_does_not_create_a_terminal_append_kink(self):
        result = extract_marked_chromatic_result(
            self._image(), (0, 0, 410, 140), np.array([18.0, 80.0]))
        vectors = np.diff(result.material.points[-4:], axis=0)
        vectors /= np.maximum(
            np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
        terminal_turn = np.degrees(np.arccos(np.clip(
            np.sum(vectors[-1] * vectors[-2]), -1.0, 1.0)))
        self.assertLess(terminal_turn, 30.0)
        # The marker remains a close terminal regularizer, not an equality.
        self.assertLess(np.linalg.norm(
            result.material.points[-1]
            - result.marker_centers_xy[3]), 1.0)

    def test_temporal_gate_does_not_propagate_marker_identity_jump(self):
        image = self._image()
        previous_left = extract_marked_chromatic_result(
            image, (0, 0, 410, 140), np.array([18.0, 80.0]))
        previous_right = extract_marked_chromatic_result(
            image, (0, 0, 410, 140), np.array([18.0, 80.0]))
        jumped = previous_left.marker_centers_xy.copy()
        jumped[0] += [45.0, 35.0]
        current_left = replace(previous_left, marker_centers_xy=jumped.copy())
        current_right = replace(previous_right, marker_centers_xy=jumped.copy())
        refined_left, refined_right = refine_marked_stereo_pair(
            image, image, current_left, current_right,
            (0, 0, 410, 140), (0, 0, 410, 140), 0.0,
            previous_left_result=previous_left,
            previous_right_result=previous_right,
            temporal_marker_max_displacement_px=18.0)
        self.assertFalse(refined_left.marker_observed[0])
        self.assertFalse(refined_right.marker_observed[0])
        self.assertTrue(np.all(np.isnan(refined_left.marker_centers_xy[0])))
        self.assertTrue(np.all(np.isnan(refined_right.marker_centers_xy[0])))

        # A correct observation on the next frame must be accepted rather than
        # compared forever against the stale pre-jump coordinate.
        recovered_left, recovered_right = refine_marked_stereo_pair(
            image, image, previous_left, previous_right,
            (0, 0, 410, 140), (0, 0, 410, 140), 0.0,
            previous_left_result=refined_left,
            previous_right_result=refined_right,
            temporal_marker_max_displacement_px=18.0)
        self.assertTrue(recovered_left.marker_observed[0])
        self.assertTrue(recovered_right.marker_observed[0])
        np.testing.assert_allclose(
            recovered_left.marker_centers_xy[0],
            previous_left.marker_centers_xy[0])

    def test_good_eye_recovers_other_eye_after_identity_snap(self):
        image = self._image()
        previous_left = extract_marked_chromatic_result(
            image, (0, 0, 410, 140), np.array([18.0, 80.0]))
        previous_right = extract_marked_chromatic_result(
            image, (0, 0, 410, 140), np.array([18.0, 80.0]))
        jumped = previous_left.marker_centers_xy.copy()
        jumped[2] += [45.0, 35.0]
        bad_left = replace(previous_left, marker_centers_xy=jumped)

        refined_left, refined_right = refine_marked_stereo_pair(
            image, image, bad_left, previous_right,
            (0, 0, 410, 140), (0, 0, 410, 140), 0.0,
            previous_left_result=previous_left,
            previous_right_result=previous_right,
            temporal_marker_max_displacement_px=18.0)
        self.assertTrue(refined_left.marker_observed[2])
        self.assertTrue(refined_right.marker_observed[2])
        self.assertLess(np.linalg.norm(
            refined_left.marker_centers_xy[2]
            - previous_left.marker_centers_xy[2]), 5.0)

    def test_epipolar_assignment_overrides_confident_wrong_eye_identities(self):
        image = self._image()
        left = extract_marked_chromatic_result(
            image, (0, 0, 410, 140), np.array([18.0, 80.0]))
        right = extract_marked_chromatic_result(
            image, (0, 0, 410, 140), np.array([18.0, 80.0]))
        expected = right.marker_centers_xy.copy()
        wrong_centers = expected.copy()
        wrong_widths = right.marker_widths_px.copy()
        wrong_confidence = np.full(4, 0.90)
        # Reproduce the real failure: IDs 1 and 2 occupy marker 1, while the
        # actual marker 2 is confidently labeled as marker 3.
        wrong_centers[2] = wrong_centers[1] + [2.0, 1.0]
        wrong_centers[3] = expected[2]
        wrong_widths[2] = wrong_widths[1]
        wrong_widths[3] = right.marker_widths_px[2]
        right = replace(
            right, marker_centers_xy=wrong_centers,
            marker_widths_px=wrong_widths,
            marker_confidence=wrong_confidence)

        _, refined_right = refine_marked_stereo_pair(
            image, image, left, right,
            (0, 0, 410, 140), (0, 0, 410, 140), 0.0)
        self.assertTrue(np.all(refined_right.marker_observed))
        np.testing.assert_allclose(
            refined_right.marker_centers_xy, expected, atol=1.0)
        self.assertLess(np.linalg.norm(
            refined_right.material.points[-1]
            - refined_right.marker_centers_xy[3]), 3.0)

    def test_independent_mask_excludes_disconnected_weak_color(self):
        image = self._image()
        # This patch has enough chroma to pass the former 0.025 expansion but
        # is not attached to strong catheter paint.
        image[105:125, 30:70] = (95, 72, 62)
        result = extract_marked_chromatic_result(
            image, (0, 0, 410, 140), np.array([18.0, 80.0]))
        self.assertEqual(int(np.count_nonzero(result.material.mask[105:125, 30:70])), 0)

    def test_workspace_mask_excludes_colored_sheath_outside_workspace(self):
        image = self._image()
        # A saturated blue sheath lies above the workspace boundary but inside
        # the rectangular image crop.
        cv2.rectangle(image, (5, 5), (80, 35), (100, 20, 15), -1)
        workspace = np.zeros(image.shape[:2], dtype=np.uint8)
        workspace[40:, :] = 255
        result = extract_marked_chromatic_result(
            image, (0, 0, 410, 140), np.array([18.0, 80.0]),
            workspace_mask=workspace)
        self.assertEqual(int(np.count_nonzero(result.material.mask[:40])), 0)
        self.assertTrue(np.all(result.material.points[:, 1] >= 40.0))

    def test_approximate_spacing_is_only_an_identity_score(self):
        candidates = [
            {"s_px": value, "width_px": width,
             "center_xy": np.array([value, 0.0]), "support": 10.0}
            for value, width in ((80.0, 8.0), (103.0, 5.0),
                                 (171.0, 5.0), (245.0, 14.0))]
        centers, _, confidence, observed = decode_marker_candidates(
            candidates, path_length_px=260.0)
        # The deliberately non-metric input coordinates are returned unchanged;
        # expected millimetre spacings merely identify the ordered observations.
        np.testing.assert_allclose(centers[:, 0], [80.0, 103.0, 171.0, 245.0])
        self.assertTrue(np.all(observed))
        self.assertTrue(np.all(confidence > 0.0))

    def test_marker_correspondences_are_soft_disparity_anchors(self):
        K = np.array([[500.0, 0.0, 320.0],
                      [0.0, 500.0, 240.0],
                      [0.0, 0.0, 1.0]])
        baseline = 0.12
        parameter = np.linspace(0.0, 1.0, 120)
        points = np.column_stack([
            0.06 * parameter,
            0.03 * np.sin(parameter * np.pi),
            0.52 + 0.03 * parameter])
        left = project_camera_points(points, K, baseline, right=False)
        right = project_camera_points(points, K, baseline, right=True)
        marker_indices = np.array([25, 45, 75, 112])
        mask = np.ones((10, 10), np.uint8)
        result = reconstruct_disparity_anchored(
            Centerline(left, (0, 0, 10, 10), mask, 1.0),
            Centerline(right, (0, 0, 10, 10), mask, 1.0),
            K, baseline, points[0], points[-1], n_samples=96,
            marker_left_px=left[marker_indices],
            marker_right_px=right[marker_indices],
            marker_confidence_left=np.ones(4),
            marker_confidence_right=np.ones(4))
        self.assertEqual(result["marker_anchor_count"], 4)
        self.assertLess(result["reprojection_p95_px"], 1.0)


if __name__ == "__main__":
    unittest.main()
