import unittest

from UI import VisualExperimentUI


class NodeCentricUITests(unittest.TestCase):
    def test_100_by_100_field_exposes_10000_distinct_node_positions(self):
        ui = VisualExperimentUI()
        ui._compose()
        layer = ui.spec.layers[0]
        points = ui._field_node_positions(layer, ui.field_rects[layer.id])
        self.assertEqual(len(points), 10000)
        self.assertEqual(len({tuple(map(int, point)) for point in points}), 10000)

    def test_branch_creates_downstream_field_without_movable_layer_card(self):
        ui = VisualExperimentUI()
        first = ui.spec.layers[0]
        ui.selected_kind = "layer"
        ui.selected_id = first.id
        ui._add_branch()
        ui._compose()
        second = ui.spec.layers[-1]
        self.assertIn(first.id, ui.field_rects)
        self.assertIn(second.id, ui.field_rects)
        self.assertLess(ui.field_rects[first.id][1], ui.field_rects[second.id][1])


if __name__ == "__main__":
    unittest.main()
