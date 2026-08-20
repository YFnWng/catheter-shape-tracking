import unittest
from contextlib import nullcontext

import cv2
import numpy as np

from shape_tracking.sam_segmentation import (
    AutomaticPromptResult,
    PromptSet,
    Sam2CatheterSegmenter,
    SamMaterialResult,
    _automatic_prompts,
    material_centerline_from_mask,
)
from shape_tracking.image_sequence import (
    _BackgroundPrefetch,
    _centerline_prompt,
    _mask_effective_width,
    _stereo_guided_retry_result,
)
from shape_tracking.materials import MaterialCenterline
from shape_tracking.segmentation import Centerline


class SamSegmentationUtilitiesTests(unittest.TestCase):
    @staticmethod
    def _synthetic_image(width=340):
        image = np.full((180, width, 3), 210, dtype=np.uint8)
        points = np.array(
            [[25, 100], [width // 3, 90],
             [2 * width // 3, 70], [width - 40, 45]], np.int32)
        cv2.polylines(image, [points[:3]], False, (75, 5, 5), 9, cv2.LINE_AA)
        cv2.line(image, tuple(points[2]), tuple(points[3]),
                 (255, 170, 20), 9, cv2.LINE_AA)
        cv2.circle(image, tuple(points[3]), 7, (0, 220, 255), -1)
        return image, points

    def test_automatic_prompts_and_mask_centerline(self):
        image, points = self._synthetic_image()
        roi = (0, 0, 340, 180)
        prompt, seed, tip = _automatic_prompts(
            image, roi, np.array([15.0, 100.0]))
        self.assertGreaterEqual(len(prompt.positive_xy), 3)
        self.assertGreater(np.count_nonzero(seed), 100)
        self.assertIsNotNone(tip)
        shaft_direction = points[-1].astype(float) - points[-2].astype(float)
        shaft_direction /= np.linalg.norm(shaft_direction)
        self.assertGreater(np.dot(tip - points[-1], shaft_direction), 2.0)
        material = material_centerline_from_mask(
            image, roi, seed, np.array([15.0, 100.0]), tip)
        self.assertIsNotNone(material)
        self.assertGreater(len(material), 100)
        self.assertLess(np.linalg.norm(material.points[0] - points[0]), 15.0)
        self.assertLess(np.linalg.norm(material.points[-1] - points[-1]), 15.0)
        np.testing.assert_allclose(material.points[-1], tip)

    def test_prepare_accepts_precomputed_automatic_prompt(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        prompt = PromptSet(
            box_xyxy=np.array([10.0, 10.0, 100.0, 70.0]),
            positive_xy=np.array([[15.0, 40.0], [95.0, 30.0]]),
            negative_xy=np.empty((0, 2)))
        seed = np.zeros((80, 120), dtype=np.uint8)
        cv2.line(seed, (15, 40), (95, 30), 255, 5)
        automatic = AutomaticPromptResult(
            prompt=prompt, seed=seed, tip=np.array([95.0, 30.0]))
        prepared = Sam2CatheterSegmenter._prepare(
            image, (0, 0, 120, 80), np.array([15.0, 40.0]),
            None, automatic)
        self.assertIs(prepared.prompt, prompt)
        np.testing.assert_array_equal(prepared.seed, seed)
        np.testing.assert_allclose(prepared.tip, automatic.tip)

    def test_temporal_prompt_replaces_current_color_prompt(self):
        image = np.zeros((80, 160, 3), dtype=np.uint8)
        temporal = PromptSet(
            box_xyxy=np.array([10.0, 15.0, 95.0, 70.0]),
            positive_xy=np.array([[15.0, 45.0], [90.0, 35.0]]),
            negative_xy=np.empty((0, 2)), source="temporal_previous_valid")
        color = PromptSet(
            box_xyxy=np.array([10.0, 10.0, 150.0, 75.0]),
            positive_xy=np.array([[15.0, 45.0], [145.0, 30.0]]),
            negative_xy=np.empty((0, 2)), source="automatic_color_seed")
        seed = np.zeros((80, 160), dtype=np.uint8)
        cv2.line(seed, (15, 45), (145, 30), 255, 5)
        automatic = AutomaticPromptResult(
            prompt=color, seed=seed, tip=np.array([145.0, 30.0]))
        prepared = Sam2CatheterSegmenter._prepare(
            image, (0, 0, 160, 80), np.array([15.0, 45.0]),
            temporal, automatic)
        self.assertIs(prepared.prompt, temporal)
        np.testing.assert_array_equal(prepared.seed, seed)

    def test_background_prefetch_preserves_order(self):
        prefetch = _BackgroundPrefetch(iter(range(12)), max_items=3)
        try:
            self.assertEqual(list(prefetch), list(range(12)))
        finally:
            prefetch.close()

    def test_batch_uses_one_encoder_call_for_two_images(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def inference_mode():
                return nullcontext()

            @staticmethod
            def autocast(*_args, **_kwargs):
                return nullcontext()

        class FakePredictor:
            def __init__(self):
                self.encoder_calls = 0
                self.images = []

            def set_image_batch(self, images):
                self.encoder_calls += 1
                self.images = images

            def predict_batch(self, **_kwargs):
                masks = []
                for image in self.images:
                    foreground = np.max(image, axis=2) - np.min(image, axis=2) > 35
                    masks.append(np.repeat(foreground[None], 3, axis=0))
                ious = [np.array([0.9, 0.8, 0.7]) for _ in self.images]
                logits = [np.empty((3, 1, 1)) for _ in self.images]
                return masks, ious, logits

        left, _ = self._synthetic_image(340)
        right, _ = self._synthetic_image(300)
        segmenter = object.__new__(Sam2CatheterSegmenter)
        segmenter.torch = FakeTorch()
        segmenter.device = "cpu"
        segmenter.predictor = FakePredictor()
        segmenter.last_timing_s = {}
        results = segmenter.segment_batch(
            [left, right], [(0, 0, 340, 180), (0, 0, 300, 180)],
            [np.array([15.0, 100.0]), np.array([15.0, 100.0])])
        self.assertEqual(len(results), 2)
        self.assertEqual(segmenter.predictor.encoder_calls, 1)
        self.assertIn("sam_image_encoder", segmenter.last_timing_s)
        self.assertIn("sam_prompt_decoder", segmenter.last_timing_s)
        self.assertIsNotNone(segmenter._postprocess_executor)
        segmenter.close()

    def test_stereo_retry_uses_colored_ridge_not_broad_mask_skeleton(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[15:85, 10:190] = 255
        points = np.column_stack([
            np.full(161, 100.0),
            np.linspace(15.0, 84.0, 161),
        ])
        material = MaterialCenterline(
            centerline=Centerline(points, (0, 0, 200, 100), mask, 30.0),
            distal_boundary_index=80,
            distal_boundary_fraction=0.5,
            boundary_confidence=1.0,
            boundary_contrast=1.0,
            material_valid=True,
            brightness_profile=np.ones(len(points)),
        )
        result = SamMaterialResult(
            material=material,
            prompt=PromptSet(
                box_xyxy=np.array([0.0, 0.0, 199.0, 99.0]),
                positive_xy=points[::40],
                negative_xy=np.empty((0, 2)),
                source="stereo_guided_retry"),
            sam_iou=0.9,
            selection_score=1.0,
            seed_recall=1.0,
            mask_area_px=int(np.count_nonzero(mask)),
            yellow_tip_xy=None,
        )
        source_points = points.copy()
        source_points[:, 0] -= 15.0
        source_material = MaterialCenterline(
            centerline=Centerline(
                source_points, (0, 0, 200, 100), mask, 5.0),
            distal_boundary_index=80,
            distal_boundary_fraction=0.5,
            boundary_confidence=1.0,
            boundary_contrast=1.0,
            material_valid=True,
            brightness_profile=np.ones(len(source_points)),
        )
        source = SamMaterialResult(
            material=source_material, prompt=result.prompt, sam_iou=0.9,
            selection_score=1.0, seed_recall=1.0,
            mask_area_px=int(np.count_nonzero(mask)), yellow_tip_xy=None)
        image = np.full((100, 200, 3), 180, dtype=np.uint8)
        # The true blue ridge is displaced from the constant-shift guide.
        cv2.line(image, (110, 15), (110, 84), (200, 80, 20), 7)
        restricted = _stereo_guided_retry_result(
            result, source=source, target_image=image,
            horizontal_shift_px=10.0,
            reference_width_px=10.0, maximum_width_px=28.0)
        self.assertLess(restricted.mask_area_px, result.mask_area_px / 2)
        self.assertLess(_mask_effective_width(restricted), 18.0)
        self.assertLess(
            np.median(np.abs(restricted.material.points[:, 0] - 110.0)), 2.0)

    def test_temporal_prompt_covers_entire_previous_centerline(self):
        mask = np.ones((100, 200), dtype=np.uint8) * 255
        points = np.array([[20.0, 20.0], [80.0, 45.0], [170.0, 80.0]])
        material = MaterialCenterline(
            centerline=Centerline(points, (0, 0, 200, 100), mask, 4.0),
            distal_boundary_index=1,
            distal_boundary_fraction=0.5,
            boundary_confidence=1.0,
            boundary_contrast=1.0,
            material_valid=True,
            brightness_profile=np.ones(len(points)))
        result = SamMaterialResult(
            material=material,
            prompt=PromptSet(
                box_xyxy=np.array([0.0, 0.0, 10.0, 10.0]),
                positive_xy=np.array([[5.0, 5.0]]),
                negative_xy=np.empty((0, 2))),
            sam_iou=0.9, selection_score=1.0, seed_recall=1.0,
            mask_area_px=int(np.count_nonzero(mask)), yellow_tip_xy=None)
        prompt = _centerline_prompt(
            result, (0, 0, 200, 100), "temporal_previous_valid")
        self.assertEqual(prompt.source, "temporal_previous_valid")
        self.assertLessEqual(prompt.box_xyxy[0], np.min(points[:, 0]))
        self.assertGreaterEqual(prompt.box_xyxy[2], np.max(points[:, 0]))
        self.assertTrue(np.any(np.all(prompt.positive_xy == points[0], axis=1)))
        self.assertTrue(np.any(np.all(prompt.positive_xy == points[-1], axis=1)))

    def test_temporal_prompt_anchors_endpoint_when_yellow_tip_is_missing(self):
        image, points = self._synthetic_image()
        # Replace the yellow marker with cyan shaft colour. Automatic prompting
        # still sees the shaft but has no explicit tip observation.
        cv2.circle(image, tuple(points[-1]), 9, (255, 170, 20), -1)
        prompt = PromptSet(
            box_xyxy=np.array([10.0, 20.0, 320.0, 130.0]),
            positive_xy=points.astype(np.float64),
            negative_xy=np.empty((0, 2)),
            source="temporal_previous_valid")
        prepared = Sam2CatheterSegmenter._prepare(
            image, (0, 0, 340, 180), np.array([15.0, 100.0]), prompt)
        self.assertIsNotNone(prepared.tip)
        np.testing.assert_allclose(prepared.tip, points[-1])


if __name__ == "__main__":
    unittest.main()
