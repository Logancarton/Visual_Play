import unittest

import numpy as np

from neural_field import SpatialNeuronField, SynapseProjection, VisualNeuronPathway
from vision_input import VisualInputExtractor


class SpatialNeuronFieldTests(unittest.TestCase):
    def test_field_has_stable_spatial_identity_for_every_neuron(self):
        field = SpatialNeuronField(3, 4)
        self.assertEqual(field.neuron_count, 12)
        self.assertEqual(field.positions.shape, (12, 2))
        self.assertTrue(np.array_equal(field.neuron_ids, np.arange(12)))
        self.assertTrue(np.all(field.positions >= 0.0))
        self.assertTrue(np.all(field.positions <= 1.0))

    def test_valid_drive_changes_only_field_state_and_bad_shape_fails_safely(self):
        field = SpatialNeuronField(2, 2, threshold=0.1, leak=0.0)
        field.step(np.array([[0.0, 0.5], [1.0, 0.0]], dtype=np.float32))
        before = field.snapshot()

        with self.assertRaises(ValueError):
            field.step(np.zeros((3, 3), dtype=np.float32))

        np.testing.assert_array_equal(field.potential, before.potential)
        np.testing.assert_array_equal(field.activity, before.activity)

    def test_reset_clears_neural_state_without_changing_identity(self):
        field = SpatialNeuronField(2, 3, threshold=0.1, leak=0.0)
        ids = field.neuron_ids.copy()
        positions = field.positions.copy()
        field.step(np.ones((2, 3), dtype=np.float32))
        field.reset()

        self.assertTrue(np.all(field.potential == 0.0))
        self.assertTrue(np.all(field.activity == 0.0))
        np.testing.assert_array_equal(field.neuron_ids, ids)
        np.testing.assert_array_equal(field.positions, positions)


class SynapseProjectionTests(unittest.TestCase):
    def test_one_to_one_projection_preserves_neuron_correspondence(self):
        projection = SynapseProjection.one_to_one(4, 4, weight=0.5)
        source = np.array([0.0, 0.2, 0.8, 1.0], dtype=np.float32)
        drive = projection.propagate(source)
        np.testing.assert_allclose(drive, source * 0.5)

        targets, weights = projection.outgoing(2)
        np.testing.assert_array_equal(targets, np.array([2], dtype=np.int32))
        np.testing.assert_allclose(weights, np.array([0.5], dtype=np.float32))

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


class VisualNeuronPathwayTests(unittest.TestCase):
    def test_brightness_reaches_next_depth_through_explicit_synapses(self):
        pathway = VisualNeuronPathway(
            rows=2,
            cols=3,
            threshold=0.0,
            leak=1.0,
            projection_weight=1.0,
        )
        brightness = np.array(
            [[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]],
            dtype=np.float32,
        )
        result = pathway.process(brightness)

        np.testing.assert_allclose(result.input_activity, brightness)
        np.testing.assert_allclose(result.downstream_activity, brightness)
        self.assertEqual(pathway.neuron_count, 12)
        self.assertEqual(pathway.synapse_count, 6)


class VisualInputTests(unittest.TestCase):
    def test_extractor_returns_only_normalized_spatial_brightness(self):
        extractor = VisualInputExtractor(rows=2, cols=3, mirror=False)
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[:, 3:, :] = 255
        result = extractor.extract(frame)
        self.assertEqual(result.brightness.shape, (2, 3))
        self.assertTrue(np.all(result.brightness >= 0.0))
        self.assertTrue(np.all(result.brightness <= 1.0))


if __name__ == "__main__":
    unittest.main()
