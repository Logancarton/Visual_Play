import unittest

from experiment_spec import ExperimentSpec
from graph_layout import (
    apply_grid,
    field_depths,
    field_world_positions,
    grid_positions,
    plane_node_positions,
    project_points_3d,
    relaxed_positions,
)


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

    def test_normal_chain_advances_on_z_not_x(self):
        spec = ExperimentSpec.default()
        first = spec.layers[0]
        second = spec.add_layer(name="Second")
        spec.add_connection(first.id, second.id)

        world = field_world_positions(spec)
        self.assertEqual(world[first.id][0], 0.0)
        self.assertEqual(world[second.id][0], 0.0)
        self.assertGreater(world[second.id][2], world[first.id][2])

    def test_explicit_branches_share_depth_and_split_laterally(self):
        spec = ExperimentSpec.default()
        first = spec.layers[0]
        left = spec.add_layer(name="Left")
        right = spec.add_layer(name="Right")
        spec.add_connection(first.id, left.id)
        spec.add_connection(first.id, right.id)

        world = field_world_positions(spec)
        self.assertEqual(world[left.id][2], world[right.id][2])
        self.assertNotEqual(world[left.id][0], world[right.id][0])

    def test_100_by_100_plane_projects_10000_distinct_node_positions(self):
        world = plane_node_positions(100, 100, (0.0, 0.0, 0.0))
        screen, _ = project_points_3d(
            world,
            screen_center=(680, 292),
            yaw_degrees=32.0,
            pitch_degrees=-16.0,
        )

        self.assertEqual(world.shape, (10000, 3))
        self.assertEqual(screen.shape, (10000, 2))
        self.assertEqual(len({tuple(point) for point in screen}), 10000)


if __name__ == "__main__":
    unittest.main()
