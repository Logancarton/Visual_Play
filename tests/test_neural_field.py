import unittest

import numpy as np

from neural_field import (
    CardinalDirectionalFlowField,
    NeuronSignalCascade,
    RetinalSignalPathway,
    SpikingNeuronField,
    SpatialNeuronField,
    SynapseProjection,
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

    def test_projection_preserves_graded_propagation_and_owns_delays(self):
        projection = SynapseProjection(
            source_count=2,
            target_count=2,
            source_indices=[0, 1],
            target_indices=[1, 0],
            weights=[0.5, -0.25],
            delays_ms=[8.0, 4.0],
        )
        drive = projection.propagate(np.array([0.4, 0.8], dtype=np.float32))
        np.testing.assert_allclose(drive, np.array([-0.2, 0.2], dtype=np.float32))
        np.testing.assert_array_equal(projection.delays_ms, [8.0, 4.0])


class SpikingNeuronFieldTests(unittest.TestCase):
    def test_subthreshold_current_accumulates_then_emits_timestamped_spike(self):
        field = SpikingNeuronField(
            1,
            1,
            membrane_tau_ms=10.0,
            threshold=0.5,
            refractory_ms=5.0,
            input_gain=2.0,
        )
        drive = np.array([1.0], dtype=np.float32)

        first = field.step(drive, timestamp_ms=1.0, dt_ms=1.0)
        self.assertEqual(first.count, 0)
        self.assertAlmostEqual(float(field.potential[0]), 0.2, places=6)

        second = field.step(drive, timestamp_ms=2.0, dt_ms=1.0)
        self.assertEqual(second.count, 0)
        self.assertAlmostEqual(float(field.potential[0]), 0.38, places=6)

        third = field.step(drive, timestamp_ms=3.0, dt_ms=1.0)
        np.testing.assert_array_equal(third.neuron_ids, [0])
        np.testing.assert_array_equal(third.timestamps_ms, [3.0])
        self.assertEqual(float(field.potential[0]), field.reset_potential)
        self.assertEqual(float(field.last_spike_ms[0]), 3.0)
        np.testing.assert_array_equal(field.adaptation, 0.0)
        np.testing.assert_array_equal(field.sustaining_current, 0.0)

    def test_refractory_period_prevents_immediate_refiring(self):
        field = SpikingNeuronField(
            1,
            1,
            membrane_tau_ms=1.0,
            threshold=0.5,
            refractory_ms=5.0,
            input_gain=1.0,
        )
        first = field.step(
            np.array([1.0], dtype=np.float32),
            timestamp_ms=1.0,
            dt_ms=1.0,
        )
        self.assertEqual(first.count, 1)

        refractory = field.step(
            np.array([5.0], dtype=np.float32),
            timestamp_ms=2.0,
            dt_ms=1.0,
        )
        self.assertEqual(refractory.count, 0)
        self.assertEqual(float(field.potential[0]), field.reset_potential)
        self.assertEqual(float(field.last_spike_ms[0]), 1.0)
        self.assertEqual(float(field.refractory_until_ms[0]), 6.0)


class NeuronSignalCascadeTests(unittest.TestCase):
    @staticmethod
    def _cascade() -> NeuronSignalCascade:
        upstream = SpikingNeuronField(
            1,
            1,
            membrane_tau_ms=1.0,
            threshold=0.5,
            refractory_ms=20.0,
            input_gain=1.0,
        )
        downstream = SpikingNeuronField(
            1,
            3,
            membrane_tau_ms=10.0,
            threshold=4.0,
            refractory_ms=5.0,
            input_gain=1.0,
        )
        projection = SynapseProjection(
            source_count=1,
            target_count=3,
            source_indices=[0, 0],
            target_indices=[0, 1],
            weights=[0.7, -0.4],
            delays_ms=[8.0, 8.0],
        )
        return NeuronSignalCascade(
            upstream,
            downstream,
            projection,
            timestep_ms=1.0,
        )

    def test_signed_synapses_arrive_only_at_connected_targets_after_delay(self):
        cascade = self._cascade()
        cascade.process(np.array([1.0], dtype=np.float32), timestamp_ms=99.0)
        spike = cascade.process(
            np.array([0.0], dtype=np.float32), timestamp_ms=100.0
        )
        np.testing.assert_array_equal(spike.upstream_spikes.neuron_ids, [0])
        np.testing.assert_array_equal(spike.upstream_spikes.timestamps_ms, [100.0])

        before = cascade.process(
            np.array([0.0], dtype=np.float32), timestamp_ms=107.0
        )
        np.testing.assert_allclose(before.downstream.potential, 0.0, atol=1e-7)

        arrived = cascade.process(
            np.array([0.0], dtype=np.float32), timestamp_ms=108.0
        )
        self.assertAlmostEqual(float(arrived.downstream.potential[0]), 0.7, places=6)
        self.assertAlmostEqual(float(arrived.downstream.potential[1]), -0.4, places=6)
        self.assertEqual(float(arrived.downstream.potential[2]), 0.0)

    def test_bad_input_and_backward_time_do_not_corrupt_cascade_state(self):
        cascade = self._cascade()
        cascade.process(np.array([1.0], dtype=np.float32), timestamp_ms=0.0)
        cascade.process(np.array([0.0], dtype=np.float32), timestamp_ms=1.0)
        before = cascade.snapshot()
        buffer_before = cascade.delay_buffer.copy()
        held_before = cascade._held_external_drive.copy()
        clock_before = cascade._neural_time_ms

        with self.assertRaises(ValueError):
            cascade.process(np.zeros(2, dtype=np.float32), timestamp_ms=2.0)
        with self.assertRaises(ValueError):
            cascade.process(np.zeros(1, dtype=np.float32), timestamp_ms=0.5)

        after = cascade.snapshot()
        np.testing.assert_array_equal(after.upstream.potential, before.upstream.potential)
        np.testing.assert_array_equal(
            after.downstream.potential, before.downstream.potential
        )
        np.testing.assert_array_equal(
            after.upstream.last_spike_ms, before.upstream.last_spike_ms
        )
        np.testing.assert_array_equal(cascade.delay_buffer, buffer_before)
        np.testing.assert_array_equal(cascade._held_external_drive, held_before)
        self.assertEqual(cascade._neural_time_ms, clock_before)


class HorizontalDirectionalFlowFieldTests(unittest.TestCase):
    def test_left_to_right_positive_sequence_activates_only_rightward_field(self):
        field = CardinalDirectionalFlowField(
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
        field = CardinalDirectionalFlowField(
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
        field = CardinalDirectionalFlowField(
            1, 3, trace_tau_ms=100.0, flow_gain=2.0
        )
        off = np.zeros((1, 3), dtype=np.float32)
        simultaneous = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
        field.process(simultaneous, off, timestamp_ms=0.0)
        result = field.process(simultaneous, off, timestamp_ms=20.0)
        np.testing.assert_allclose(result.rightward_activity, 0.0, atol=1e-7)
        np.testing.assert_allclose(result.leftward_activity, 0.0, atol=1e-7)

    def test_dark_change_can_drive_rightward_direction(self):
        field = CardinalDirectionalFlowField(
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
        field = CardinalDirectionalFlowField(
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
        field = CardinalDirectionalFlowField(1, 3)
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
        field = CardinalDirectionalFlowField(
            4, 2, trace_tau_ms=100.0, flow_gain=2.0
        )
        off = np.zeros((4, 2), dtype=np.float32)
        field.process(
            np.array(
                [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                dtype=np.float32,
            ),
            off,
            timestamp_ms=0.0,
        )
        result = field.process(
            np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                dtype=np.float32,
            ),
            off,
            timestamp_ms=20.0,
        )
        self.assertGreater(float(result.downward_activity[1, 0]), 0.0)
        np.testing.assert_allclose(result.upward_activity, 0.0, atol=1e-7)

    def test_bottom_to_top_positive_sequence_activates_only_upward_field(self):
        field = CardinalDirectionalFlowField(
            4, 2, trace_tau_ms=100.0, flow_gain=2.0
        )
        off = np.zeros((4, 2), dtype=np.float32)
        field.process(
            np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                dtype=np.float32,
            ),
            off,
            timestamp_ms=0.0,
        )
        result = field.process(
            np.array(
                [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                dtype=np.float32,
            ),
            off,
            timestamp_ms=20.0,
        )
        self.assertGreater(float(result.upward_activity[0, 0]), 0.0)
        np.testing.assert_allclose(result.downward_activity, 0.0, atol=1e-7)

    def test_synchronous_vertical_neighbor_change_cancels_both_directions(self):
        field = CardinalDirectionalFlowField(
            3, 2, trace_tau_ms=100.0, flow_gain=2.0
        )
        off = np.zeros((3, 2), dtype=np.float32)
        simultaneous = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32
        )
        field.process(simultaneous, off, timestamp_ms=0.0)
        result = field.process(simultaneous, off, timestamp_ms=20.0)
        np.testing.assert_allclose(result.downward_activity, 0.0, atol=1e-7)
        np.testing.assert_allclose(result.upward_activity, 0.0, atol=1e-7)

    def test_dark_change_can_drive_downward_direction(self):
        field = CardinalDirectionalFlowField(
            4, 2, trace_tau_ms=100.0, flow_gain=2.0
        )
        on = np.zeros((4, 2), dtype=np.float32)
        field.process(
            on,
            np.array(
                [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                dtype=np.float32,
            ),
            timestamp_ms=0.0,
        )
        result = field.process(
            on,
            np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                dtype=np.float32,
            ),
            timestamp_ms=20.0,
        )
        self.assertGreater(float(result.downward_activity[1, 0]), 0.0)
        np.testing.assert_allclose(result.upward_activity, 0.0, atol=1e-7)

    def test_axis_comparisons_do_not_own_duplicate_temporal_traces(self):
        field = CardinalDirectionalFlowField(3, 2)
        self.assertEqual(field.on_trace.shape, (3, 2))
        self.assertEqual(field.off_trace.shape, (3, 2))
        self.assertFalse(hasattr(field.horizontal_flow, "on_trace"))
        self.assertFalse(hasattr(field.horizontal_flow, "off_trace"))
        self.assertFalse(hasattr(field.vertical_flow, "on_trace"))
        self.assertFalse(hasattr(field.vertical_flow, "off_trace"))


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
        self.assertEqual(result.positive_spike_cascade.upstream_spikes.count, 0)
        self.assertEqual(result.positive_spike_cascade.downstream_spikes.count, 0)
        np.testing.assert_allclose(
            result.positive_spike_cascade.upstream.potential, 0.0, atol=1e-7
        )
        np.testing.assert_allclose(
            result.positive_spike_cascade.downstream.potential, 0.0, atol=1e-7
        )
        self.assertEqual(pathway.neuron_count, 42)
        self.assertEqual(pathway.spiking_neuron_count, 12)

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

    def test_live_brightness_sequence_reaches_upward_branch_only(self):
        pathway = RetinalSignalPathway(
            4,
            2,
            baseline_tau_ms=1000.0,
            response_gain=3.0,
            flow_trace_tau_ms=100.0,
            flow_gain=4.0,
        )
        neutral = np.full((4, 2), 0.5, dtype=np.float32)
        bottom = neutral.copy()
        top = neutral.copy()
        bottom[1, 0] = 1.0
        top[0, 0] = 1.0
        pathway.process(neutral, timestamp_ms=0.0)
        pathway.process(bottom, timestamp_ms=20.0)
        result = pathway.process(top, timestamp_ms=40.0)
        self.assertGreater(float(result.upward_activity[0, 0]), 0.0)
        np.testing.assert_allclose(result.downward_activity, 0.0, atol=1e-7)

    def test_live_positive_variation_drives_delayed_spiking_cascade(self):
        pathway = RetinalSignalPathway(
            1,
            2,
            baseline_tau_ms=1000.0,
            response_gain=3.0,
            spike_timestep_ms=1.0,
            spike_synaptic_weight=0.7,
            spike_synaptic_delay_ms=8.0,
        )
        neutral = np.full((1, 2), 0.5, dtype=np.float32)
        bright = neutral.copy()
        bright[0, 0] = 1.0

        pathway.process(neutral, timestamp_ms=0.0)
        onset = pathway.process(bright, timestamp_ms=1.0)
        self.assertGreater(float(onset.on_activity[0, 0]), 0.0)

        accumulating = pathway.process(bright, timestamp_ms=3.0)
        self.assertGreater(
            float(accumulating.positive_spike_cascade.upstream.potential[0]),
            0.0,
        )
        self.assertEqual(accumulating.positive_spike_cascade.upstream_spikes.count, 0)

        fired = pathway.process(bright, timestamp_ms=4.0)
        np.testing.assert_array_equal(
            fired.positive_spike_cascade.upstream_spikes.neuron_ids, [0]
        )
        np.testing.assert_array_equal(
            fired.positive_spike_cascade.upstream_spikes.timestamps_ms, [4.0]
        )

        before_delay = pathway.process(bright, timestamp_ms=11.0)
        self.assertEqual(
            float(before_delay.positive_spike_cascade.downstream.potential[0]),
            0.0,
        )
        arrived = pathway.process(bright, timestamp_ms=12.0)
        self.assertAlmostEqual(
            float(arrived.positive_spike_cascade.downstream.potential[0]),
            0.7,
            places=6,
        )
        self.assertEqual(
            float(arrived.positive_spike_cascade.downstream.potential[1]),
            0.0,
        )

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
        np.testing.assert_array_equal(
            after.positive_spike_cascade.upstream.potential,
            before.positive_spike_cascade.upstream.potential,
        )
        np.testing.assert_array_equal(
            after.positive_spike_cascade.downstream.potential,
            before.positive_spike_cascade.downstream.potential,
        )

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
        np.testing.assert_array_equal(
            after.positive_spike_cascade.upstream.potential,
            before.positive_spike_cascade.upstream.potential,
        )
        np.testing.assert_array_equal(
            after.positive_spike_cascade.downstream.potential,
            before.positive_spike_cascade.downstream.potential,
        )


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
