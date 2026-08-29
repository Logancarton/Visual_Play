import unittest

from experiment_spec import ExperimentSpec
from graph_layout import apply_grid, relaxed_positions


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

        apply_grid(spec)
        self.assertNotEqual(left.x, right.x)
        self.assertLess(first.y, left.y)
        self.assertLess(left.y, deeper.y)

    def test_runtime_strength_can_change_visual_pull_without_mutating_spec_math(self):
        spec = ExperimentSpec.default()
        second = spec.add_layer(x=0.90, y=0.90)
        connection = spec.add_connection(spec.layers[0].id, second.id)
        before = (second.x, second.y)

        weak = relaxed_positions(spec, runtime_strengths={connection.id: 0.1})[second.id]
        strong = relaxed_positions(spec, runtime_strengths={connection.id: 5.0})[second.id]

        weak_move = abs(weak[0] - before[0]) + abs(weak[1] - before[1])
        strong_move = abs(strong[0] - before[0]) + abs(strong[1] - before[1])
        self.assertGreater(strong_move, weak_move)
        self.assertEqual(connection.gain, 1.0)

    def test_pinned_layer_does_not_move(self):
        spec = ExperimentSpec.default()
        layer = spec.layers[0]
        layer.pinned = True
        before = (layer.x, layer.y)
        after = relaxed_positions(spec)[layer.id]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
