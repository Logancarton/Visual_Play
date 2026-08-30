import unittest

import numpy as np

from neural_field import (
    HorizontalDirectionalFlowField,
    RetinalSignalPathway,
    SpatialNeuronField,
    SynapseProjection,
    VerticalDirectionalFlowField,
)
from vision_input import VisualInputExtractor


class SpatialNeuronFieldTests(unittest.TestCase):
    def test_field_has_stable_spatial_identity_for_every_neuron(self):
        field = SpatialNeuronField(3, 4)
        self.assertEqual(field.neuron_count, 12)
        self.assertEqual(field.positions.shape, (12, 2))
        self.assertTrue(np.array_equal(field.neuron_ids, np.arange(12)))

    def test_bad_drive_does_not_mutate_existing_state(self):
        field = SpatialNeuronField(2, 2, threshold=0.1, leak=0.0)
        field.step(np.array([[0.0, 0.5], [1.0, 0.0]], dtype=np.float32))
        before = field.snapshot()
        with self.assertRaises(ValueError):
            field.step(np.zeros((3, 3), dtype=np.float32))
        np.testing.assert_array_equal(field.potential, before.potential)
        np.testing.assert_array_equal(field.activity, before.activity)


class SynapseProjectionTests(unittest.TestCase):
    def test_sparse_synapses_sum_real_pairwise_contributions(self):
        projection = SynapseProjection(
            source_count=3,
            target_count=2,
            source_indices=[0, 1, 2],
            target_indices=[0, 0, 1],
            weights=[0.5, -0.25, 2.0],
        )
        drive = projection.propagate(np.array([1.0, 0.4, 0.25], dtype=np.float32))
        np.testing.assert_allclose(drive, np.array([0.4, 0.5], dtype=np.float32))


class HorizontalDirectionalFlowFieldTests(unittest.TestCase):
    def test_left_to_right_positive_sequence_activates_only_rightward_field(self):
        field = HorizontalDirectionalFlowField(
            1, 4, trace_tau_ms=100.0, flow_gain=2.0
        )
        off = np.zeros((1, 4), dtype=np.float32)
        field.process(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            off,
            timestamp_ms=0.0,
        )
        result = field.process(
            np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
            off,
            timestamp_ms=20.0,
        )
        self.assertGreater(float(result.rightward_activity[0, 1]), 0.0)
        np.testing.assert_allclose(result.leftward_activity, 0.0, atol=1e-7)

    def test_right_to_left_positive_sequence_activates_only_leftward_field(self):
        field = HorizontalDirectionalFlowField(
            1, 4, trace_tau_ms=100.0, flow_gain=2.0
        )
        off = np.zeros((1, 4), dtype=np.float32)
        field.process(
            np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
            off,
            timestamp_ms=0.0,
        )
        result = field.process(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            off,
            timestamp_ms=20.0,
        )
        self.assertGreater(float(result.leftward_activity[0, 0]), 0.0)
        np.testing.assert_allclose(result.rightward_activity, 0.0, atol=1e-7)

    def test_synchronous_neighbor_change_cancels_in_both_directions(self):
        field = HorizontalDirectionalFlowField(
            1, 3, trace_tau_ms=100.0, flow_gain=2.0
        )
        off = np.zeros((1, 3), dtype=np.float32)
        simultaneous = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
        field.process(simultaneous, off, timestamp_ms=0.0)
        result = field.process(simultaneous, off, timestamp_ms=20.0)
        np.testing.assert_allclose(result.rightward_activity, 0.0, atol=1e-7)
        np.testing.assert_allclose(result.leftward_activity, 0.0, atol=1e-7)

    def test_dark_change_can_drive_rightward_direction(self):
        field = HorizontalDirectionalFlowField(
            1, 4, trace_tau_ms=100.0, flow_gain=2.0
        )
        on = np.zeros((1, 4), dtype=np.float32)
        field.process(
            on,
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            timestamp_ms=0.0,
        )
        result = field.process(
            on,
            np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
            timestamp_ms=20.0,
        )
        self.assertGreater(float(result.rightward_activity[0, 1]), 0.0)
        np.testing.assert_allclose(result.leftward_activity, 0.0, atol=1e-7)

    def test_dark_change_can_drive_leftward_direction(self):
        field = HorizontalDirectionalFlowField(
            1, 4, trace_tau_ms=100.0, flow_gain=2.0
        )
        on = np.zeros((1, 4), dtype=np.float32)
        field.process(
            on,
            np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
            timestamp_ms=0.0,
        )
        result = field.process(
            on,
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            timestamp_ms=20.0,
        )
        self.assertGreater(float(result.leftward_activity[0, 0]), 0.0)
        np.testing.assert_allclose(result.rightward_activity, 0.0, atol=1e-7)

    def test_bad_shape_and_backward_time_do_not_corrupt_flow_state(self):
        field = HorizontalDirectionalFlowField(1, 3)
        zeros = np.zeros((1, 3), dtype=np.float32)
        field.process(zeros, zeros, timestamp_ms=100.0)
        before = field.snapshot()
        with self.assertRaises(ValueError):
            field.process(
                np.zeros((2, 3), dtype=np.float32),
                zeros,
                timestamp_ms=110.0,
            )
        with self.assertRaises(ValueError):
            field.process(zeros, zeros, timestamp_ms=90.0)
        after = field.snapshot()
        np.testing.assert_array_equal(
            after.rightward_activity, before.rightward_activity
        )
        np.testing.assert_array_equal(after.leftward_activity, before.leftward_activity)
        np.testing.assert_array_equal(after.on_trace, before.on_trace)
        np.testing.assert_array_equal(after.off_trace, before.off_trace)


class VerticalDirectionalFlowFieldTests(unittest.TestCase):
    def test_top_to_bottom_positive_sequence_activates_only_downward_field(self):
        field = VerticalDirectionalFlowField(
            4, 1, trace_tau_ms=100.0, flow_gain=2.0
        )
        off = np.zeros((4, 1), dtype=np.float32)
        field.process(
            np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32),
            off,
            timestamp_ms=0.0,
        )
        result = field.process(
            np.array([[0.0], [1.0], [0.0], [0.0]], dtype=np.float32),
            off,
            timestamp_ms=20.0,
        )
        self.assertGreater(float(result.downward_activity[1, 0]), 0.0)
        np.testing.assert_allclose(result.upward_activity, 0.0, atol=1e-7)

    def test_bottom_to_top_positive_sequence_activates_only_upward_field(self):
        field = VerticalDirectionalFlowField(
            4, 1, trace_tau_ms=100.0, flow_gain=2.0
        )
        off = np.zeros((4, 1), dtype=np.float32)
        field.process(
            np.array([[0.0], [1.0], [0.0], [0.0]], dtype=np.float32),
            off,
            timestamp_ms=0.0,
        )
        result = field.process(
            np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32),
            off,
            timestamp_ms=20.0,
        )
        self.assertGreater(float(result.upward_activity[0, 0]), 0.0)
        np.testing.assert_allclose(result.downward_activity, 0.0, atol=1e-7)

    def test_synchronous_vertical_neighbor_change_cancels_both_directions(self):
        field = VerticalDirectionalFlowField(
            3, 1, trace_tau_ms=100.0, flow_gain=2.0
        )
        off = np.zeros((3, 1), dtype=np.float32)
        simultaneous = np.array([[1.0], [1.0], [0.0]], dtype=np.float32)
        field.process(simultaneous, off, timestamp_ms=0.0)
        result = field.process(simultaneous, off, timestamp_ms=20.0)
        np.testing.assert_allclose(result.downward_activity, 0.0, atol=1e-7)
        np.testing.assert_allclose(result.upward_activity, 0.0, atol=1e-7)

    def test_dark_change_can_drive_downward_direction(self):
        field = VerticalDirectionalFlowField(
            4, 1, trace_tau_ms=100.0, flow_gain=2.0
        )
        on = np.zeros((4, 1), dtype=np.float32)
        field.process(
            on,
            np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32),
            timestamp_ms=0.0,
        )
        result = field.process(
            on,
            np.array([[0.0], [1.0], [0.0], [0.0]], dtype=np.float32),
            timestamp_ms=20.0,
        )
        self.assertGreater(float(result.downward_activity[1, 0]), 0.0)
        np.testing.assert_allclose(result.upward_activity, 0.0, atol=1e-7)

    def test_bad_shape_and_backward_time_do_not_corrupt_flow_state(self):
        field = VerticalDirectionalFlowField(3, 1)
        zeros = np.zeros((3, 1), dtype=np.float32)
        field.process(zeros, zeros, timestamp_ms=100.0)
        before = field.snapshot()
        with self.assertRaises(ValueError):
            field.process(
                np.zeros((3, 2), dtype=np.float32),
                zeros,
                timestamp_ms=110.0,
            )
        with self.assertRaises(ValueError):
            field.process(zeros, zeros, timestamp_ms=90.0)
        after = field.snapshot()
        np.testing.assert_array_equal(
            after.downward_activity, before.downward_activity
        )
        np.testing.assert_array_equal(after.upward_activity, before.upward_activity)
        np.testing.assert_array_equal(after.on_trace, before.on_trace)
        np.testing.assert_array_equal(after.off_trace, before.off_trace)


class RetinalSignalPathwayTests(unittest.TestCase):
    def test_first_frame_seeds_temporal_baseline_without_fabricating_variation_or_flow(self):
        pathway = RetinalSignalPathway(2, 3)
        first = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
        )
        result = pathway.process(first, timestamp_ms=100.0)
        np.testing.assert_allclose(result.baseline, first)
        self.assertTrue(np.all(result.on_activity == 0.0))
        self.assertTrue(np.all(result.off_activity == 0.0))
        self.assertTrue(np.all(result.rightward_activity == 0.0))
        self.assertTrue(np.all(result.leftward_activity == 0.0))
        self.assertTrue(np.all(result.downward_activity == 0.0))
        self.assertTrue(np.all(result.upward_activity == 0.0))
        self.assertEqual(pathway.neuron_count, 42)

    def test_each_location_splits_brighter_and_darker_change_independently(self):
        pathway = RetinalSignalPathway(1, 3, response_gain=2.0)
        pathway.process(
            np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
            timestamp_ms=0.0,
        )
        result = pathway.process(
            np.array([[0.7, 0.3, 0.5]], dtype=np.float32),
            timestamp_ms=10.0,
        )
        np.testing.assert_allclose(
            result.on_activity, [[0.4, 0.0, 0.0]], atol=1e-6
        )
        np.testing.assert_allclose(
            result.off_activity, [[0.0, 0.4, 0.0]], atol=1e-6
        )

    def test_unchanged_input_fades_as_each_local_baseline_adapts(self):
        pathway = RetinalSignalPathway(
            1,
            2,
            baseline_tau_ms=100.0,
            response_gain=1.0,
        )
        pathway.process(
            np.array([[0.0, 0.0]], dtype=np.float32),
            timestamp_ms=0.0,
        )
        early = pathway.process(
            np.array([[1.0, 0.0]], dtype=np.float32),
            timestamp_ms=10.0,
        )
        later = pathway.process(
            np.array([[1.0, 0.0]], dtype=np.float32),
            timestamp_ms=210.0,
        )
        self.assertGreater(
            float(early.on_activity[0, 0]),
            float(later.on_activity[0, 0]),
        )
        self.assertEqual(float(later.off_activity[0, 0]), 0.0)

    def test_uniform_spatial_input_has_zero_local_contrast(self):
        pathway = RetinalSignalPathway(3, 3)
        result = pathway.process(
            np.full((3, 3), 0.5, dtype=np.float32),
            timestamp_ms=0.0,
        )
        np.testing.assert_allclose(result.contrast_activity, 0.0, atol=1e-7)

    def test_bright_center_produces_local_contrast_at_same_center(self):
        pathway = RetinalSignalPathway(3, 3, contrast_gain=2.0)
        image = np.full((3, 3), 0.5, dtype=np.float32)
        image[1, 1] = 1.0
        result = pathway.process(image, timestamp_ms=0.0)
        self.assertGreater(float(result.contrast_activity[1, 1]), 0.0)
        self.assertGreater(
            float(result.contrast_activity[1, 1]),
            float(result.contrast_activity[0, 0]),
        )

    def test_live_brightness_sequence_reaches_rightward_branch_only(self):
        pathway = RetinalSignalPathway(
            1,
            4,
            baseline_tau_ms=1000.0,
            response_gain=3.0,
            flow_trace_tau_ms=100.0,
            flow_gain=4.0,
        )
        neutral = np.full((1, 4), 0.5, dtype=np.float32)
        left = neutral.copy()
        right = neutral.copy()
        left[0, 0] = 1.0
        right[0, 1] = 1.0
        pathway.process(neutral, timestamp_ms=0.0)
        pathway.process(left, timestamp_ms=20.0)
        result = pathway.process(right, timestamp_ms=40.0)
        self.assertGreater(float(result.rightward_activity[0, 1]), 0.0)
        np.testing.assert_allclose(result.leftward_activity, 0.0, atol=1e-7)

    def test_live_brightness_sequence_reaches_leftward_branch_only(self):
        pathway = RetinalSignalPathway(
            1,
            4,
            baseline_tau_ms=1000.0,
            response_gain=3.0,
            flow_trace_tau_ms=100.0,
            flow_gain=4.0,
        )
        neutral = np.full((1, 4), 0.5, dtype=np.float32)
        right = neutral.copy()
        left = neutral.copy()
        right[0, 1] = 1.0
        left[0, 0] = 1.0
        pathway.process(neutral, timestamp_ms=0.0)
        pathway.process(right, timestamp_ms=20.0)
        result = pathway.process(left, timestamp_ms=40.0)
        self.assertGreater(float(result.leftward_activity[0, 0]), 0.0)
        np.testing.assert_allclose(result.rightward_activity, 0.0, atol=1e-7)

    def test_live_brightness_sequence_reaches_downward_branch_only(self):
        pathway = RetinalSignalPathway(
            4,
            2,
            baseline_tau_ms=1000.0,
            response_gain=3.0,
            flow_trace_tau_ms=100.0,
            flow_gain=4.0,
        )
        neutral = np.full((4, 2), 0.5, dtype=np.float32)
        top = neutral.copy()
        bottom = neutral.copy()
        top[0, 0] = 1.0
        bottom[1, 0] = 1.0
        pathway.process(neutral, timestamp_ms=0.0)
        pathway.process(top, timestamp_ms=20.0)
        result = pathway.process(bottom, timestamp_ms=40.0)
        self.assertGreater(float(result.downward_activity[1, 0]), 0.0)
        np.testing.assert_allclose(result.upward_activity, 0.0, atol=1e-7)

    def test_bad_input_fails_without_mutating_retinal_state(self):
        pathway = RetinalSignalPathway(2, 2)
        pathway.process(
            np.full((2, 2), 0.5, dtype=np.float32),
            timestamp_ms=100.0,
        )
        before = pathway.snapshot()
        with self.assertRaises(ValueError):
            pathway.process(
                np.zeros((3, 3), dtype=np.float32),
                timestamp_ms=110.0,
            )
        after = pathway.snapshot()
        np.testing.assert_array_equal(after.baseline, before.baseline)
        np.testing.assert_array_equal(after.on_activity, before.on_activity)
        np.testing.assert_array_equal(after.off_activity, before.off_activity)
        np.testing.assert_array_equal(
            after.contrast_activity, before.contrast_activity
        )
        np.testing.assert_array_equal(
            after.rightward_activity, before.rightward_activity
        )
        np.testing.assert_array_equal(
            after.leftward_activity, before.leftward_activity
        )
        np.testing.assert_array_equal(
            after.downward_activity, before.downward_activity
        )
        np.testing.assert_array_equal(after.upward_activity, before.upward_activity)

    def test_backward_time_fails_without_mutating_retinal_state(self):
        pathway = RetinalSignalPathway(1, 2)
        pathway.process(
            np.array([[0.5, 0.5]], dtype=np.float32),
            timestamp_ms=100.0,
        )
        before = pathway.snapshot()
        with self.assertRaises(ValueError):
            pathway.process(
                np.array([[0.9, 0.5]], dtype=np.float32),
                timestamp_ms=90.0,
            )
        after = pathway.snapshot()
        np.testing.assert_array_equal(after.baseline, before.baseline)
        np.testing.assert_array_equal(after.on_activity, before.on_activity)
        np.testing.assert_array_equal(after.off_activity, before.off_activity)
        np.testing.assert_array_equal(
            after.contrast_activity, before.contrast_activity
        )
        np.testing.assert_array_equal(
            after.rightward_activity, before.rightward_activity
        )
        np.testing.assert_array_equal(
            after.leftward_activity, before.leftward_activity
        )
        np.testing.assert_array_equal(
            after.downward_activity, before.downward_activity
        )
        np.testing.assert_array_equal(after.upward_activity, before.upward_activity)


class VisualInputTests(unittest.TestCase):
    def test_extractor_returns_normalized_spatial_brightness(self):
        extractor = VisualInputExtractor(rows=2, cols=3, mirror=False)
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[:, 3:, :] = 255
        result = extractor.extract(frame)
        self.assertEqual(result.brightness.shape, (2, 3))
        self.assertTrue(np.all(result.brightness >= 0.0))
        self.assertTrue(np.all(result.brightness <= 1.0))


if __name__ == "__main__":
    unittest.main()
