import unittest

from experiment_spec import ExperimentSpec
from graph_layout import apply_grid, field_depths, grid_positions, relaxed_positions


class GraphLayoutTests(unittest.TestCase):
    def test_grid_spreads_branches_and_orders_depth(self):
        spec = ExperimentSpec.default()
        first = spec.layers[0]
        left = spec.add_layer(name="Left")
        right = spec.add_layer(name="Right")
        deeper = spec.add_layer(name="Deeper")
        spec.add_connection(first.id, left.id)
        spec.add_connection(first.id, right.id)
        spec.add_connection(left.id, deeper.id)

        positions = apply_grid(spec)
        self.assertNotEqual(positions[left.id][0], positions[right.id][0])
        self.assertLess(positions[first.id][1], positions[left.id][1])
        self.assertLess(positions[left.id][1], positions[deeper.id][1])
        depths = field_depths(spec)
        self.assertEqual(depths[first.id], 0)
        self.assertEqual(depths[left.id], 1)
        self.assertEqual(depths[deeper.id], 2)

    def test_runtime_strength_cannot_move_whole_fields(self):
        spec = ExperimentSpec.default()
        second = spec.add_layer(name="Second")
        connection = spec.add_connection(spec.layers[0].id, second.id)

        weak = relaxed_positions(spec, runtime_strengths={connection.id: 0.1})
        strong = relaxed_positions(spec, runtime_strengths={connection.id: 100.0})

        self.assertEqual(weak, strong)
        self.assertEqual(weak, grid_positions(spec))
        self.assertEqual(connection.gain, 1.0)

    def test_second_field_is_next_structural_depth(self):
        spec = ExperimentSpec.default()
        second = spec.add_layer(name="Second")
        spec.add_connection(spec.layers[0].id, second.id)
        positions = grid_positions(spec)
        self.assertLess(positions[spec.layers[0].id][1], positions[second.id][1])

    def test_recurrent_cycle_is_grouped_without_infinite_depth(self):
        spec = ExperimentSpec.default()
        second = spec.add_layer(name="Second")
        third = spec.add_layer(name="Third")
        spec.add_connection(spec.layers[0].id, second.id)
        spec.add_connection(second.id, third.id)
        spec.add_connection(third.id, second.id)
        depths = field_depths(spec)
        self.assertLessEqual(max(depths.values()), len(spec.layers))


if __name__ == "__main__":
    unittest.main()
