import tempfile
import unittest
from pathlib import Path

from experiment_spec import ExperimentSpec, MECHANISMS


class ExperimentSpecTests(unittest.TestCase):
    def test_default_experiment_is_brightness_to_100x100_layer(self):
        spec = ExperimentSpec.default()
        self.assertEqual(len(spec.layers), 1)
        self.assertEqual((spec.layers[0].rows, spec.layers[0].cols), (100, 100))
        self.assertEqual(spec.layers[0].unit_count, 10000)
        self.assertEqual(len(spec.connections), 1)
        self.assertEqual(spec.connections[0].source_id, "sensor:brightness")
        self.assertEqual(spec.connections[0].target_id, spec.layers[0].id)

    def test_layers_branch_without_duplicate_connection_authority(self):
        spec = ExperimentSpec.default()
        second = spec.add_layer()
        first = spec.layers[0]
        spec.add_connection(first.id, second.id)
        spec.add_connection("sensor:brightness", second.id)

        pairs = {(c.source_id, c.target_id) for c in spec.connections}
        self.assertEqual(len(pairs), len(spec.connections))

        with self.assertRaises(ValueError):
            spec.add_connection(first.id, second.id)

    def test_mechanisms_and_dimensions_are_declarative_and_bounded(self):
        spec = ExperimentSpec.default()
        layer = spec.layers[0]

        before = layer.mechanisms["stdp"]
        spec.toggle_mechanism(layer.id, "stdp")
        self.assertNotEqual(before, layer.mechanisms["stdp"])
        self.assertEqual(set(layer.mechanisms), set(MECHANISMS))

        spec.set_dimensions(layer.id, rows=10000, cols=-20)
        self.assertEqual(layer.rows, 512)
        self.assertEqual(layer.cols, 1)

    def test_delete_layer_removes_only_attached_connections(self):
        spec = ExperimentSpec.default()
        first = spec.layers[0]
        second = spec.add_layer()
        third = spec.add_layer()
        keep = spec.add_connection("sensor:brightness", third.id)
        spec.add_connection(first.id, second.id)

        self.assertTrue(spec.remove_layer(second.id))
        self.assertIsNotNone(spec.connection_by_id(keep.id))
        self.assertIsNone(spec.layer_by_id(second.id))

    def test_round_trip_save_load(self):
        spec = ExperimentSpec.default()
        layer = spec.layers[0]
        spec.toggle_mechanism(layer.id, "stdp")
        spec.set_visualization("potential")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.json"
            spec.save(path)
            loaded = ExperimentSpec.load(path)

        self.assertEqual(loaded.to_dict(), spec.to_dict())


if __name__ == "__main__":
    unittest.main()
