import unittest

import numpy as np

from neural_field import RetinalVariationPathway, SpatialNeuronField, SynapseProjection
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


class RetinalVariationPathwayTests(unittest.TestCase):
    def test_first_frame_seeds_local_baseline_without_fabricating_change(self):
        pathway = RetinalVariationPathway(2, 3)
        first = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        result = pathway.process(first, timestamp_ms=100.0)
        np.testing.assert_allclose(result.baseline, first)
        self.assertTrue(np.all(result.on_activity == 0.0))
        self.assertTrue(np.all(result.off_activity == 0.0))
        self.assertEqual(pathway.neuron_count, 12)

    def test_each_location_splits_brighter_and_darker_change_independently(self):
        pathway = RetinalVariationPathway(1, 3, response_gain=2.0)
        pathway.process(np.array([[0.5, 0.5, 0.5]], dtype=np.float32), timestamp_ms=0.0)
        result = pathway.process(
            np.array([[0.7, 0.3, 0.5]], dtype=np.float32),
            timestamp_ms=10.0,
        )
        np.testing.assert_allclose(result.on_activity, [[0.4, 0.0, 0.0]], atol=1e-6)
        np.testing.assert_allclose(result.off_activity, [[0.0, 0.4, 0.0]], atol=1e-6)

    def test_unchanged_input_fades_as_each_local_baseline_adapts(self):
        pathway = RetinalVariationPathway(
            1,
            1,
            baseline_tau_ms=100.0,
            response_gain=1.0,
        )
        pathway.process(np.array([[0.0]], dtype=np.float32), timestamp_ms=0.0)
        early = pathway.process(np.array([[1.0]], dtype=np.float32), timestamp_ms=10.0)
        later = pathway.process(np.array([[1.0]], dtype=np.float32), timestamp_ms=210.0)
        self.assertGreater(float(early.on_activity[0, 0]), float(later.on_activity[0, 0]))
        self.assertEqual(float(later.off_activity[0, 0]), 0.0)

    def test_backward_time_fails_without_mutating_retinal_state(self):
        pathway = RetinalVariationPathway(1, 1)
        pathway.process(np.array([[0.5]], dtype=np.float32), timestamp_ms=100.0)
        before = pathway.snapshot()
        with self.assertRaises(ValueError):
            pathway.process(np.array([[0.9]], dtype=np.float32), timestamp_ms=90.0)
        after = pathway.snapshot()
        np.testing.assert_array_equal(after.baseline, before.baseline)
        np.testing.assert_array_equal(after.on_activity, before.on_activity)
        np.testing.assert_array_equal(after.off_activity, before.off_activity)


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
