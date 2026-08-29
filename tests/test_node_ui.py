import unittest

from UI import VisualExperimentUI
from graph_layout import field_world_positions


class NodeCentric3DUITests(unittest.TestCase):
    def test_100_by_100_field_exposes_10000_projected_nodes(self):
        ui = VisualExperimentUI()
        ui._compose()

        first = ui.spec.layers[0]
        points = ui.field_screen_nodes[first.id]

        self.assertEqual(len(points), 10000)
        self.assertEqual(len({tuple(point) for point in points}), 10000)

    def test_add_next_field_builds_depth_chain(self):
        ui = VisualExperimentUI()
        first = ui.spec.layers[0]
        ui.selected_kind = "layer"
        ui.selected_id = first.id

        ui._control("add")
        second = ui.spec.layers[-1]
        world = field_world_positions(ui.spec)

        self.assertEqual(world[first.id][0], 0.0)
        self.assertEqual(world[second.id][0], 0.0)
        self.assertGreater(world[second.id][2], world[first.id][2])
        self.assertTrue(
            any(
                path.source_id == first.id and path.target_id == second.id
                for path in ui.spec.connections
            )
        )

    def test_explicit_branch_splits_laterally_at_same_depth(self):
        ui = VisualExperimentUI()
        first = ui.spec.layers[0]

        ui.selected_kind = "layer"
        ui.selected_id = first.id
        ui._control("branch")
        branch_one = ui.spec.layers[-1]

        ui.selected_kind = "layer"
        ui.selected_id = first.id
        ui._control("branch")
        branch_two = ui.spec.layers[-1]

        world = field_world_positions(ui.spec)
        self.assertEqual(world[branch_one.id][2], world[branch_two.id][2])
        self.assertNotEqual(world[branch_one.id][0], world[branch_two.id][0])


if __name__ == "__main__":
    unittest.main()
