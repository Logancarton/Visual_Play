"""
experiment_spec.py

Declarative experiment description for Visual_Play.

This file owns what the user wants to build: sensory sources, neural fields,
connections, mechanisms, and tunable structural parameters. It intentionally
contains no neural simulation math.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

SENSORY_SOURCES = (
    "sensor:brightness",
    "sensor:contrast",
    "sensor:motion",
    "sensor:horizontal",
    "sensor:vertical",
)

SENSORY_LABELS = {
    "sensor:brightness": "Brightness",
    "sensor:contrast": "Contrast",
    "sensor:motion": "Motion",
    "sensor:horizontal": "Horizontal",
    "sensor:vertical": "Vertical",
}

MECHANISMS = (
    "threshold",
    "leak",
    "refractory",
    "adaptation",
    "local_excitation",
    "lateral_inhibition",
    "homeostasis",
    "hebbian",
    "oja",
    "stdp",
    "structural_growth",
    "pruning",
)

CONNECTION_PATTERNS = ("one_to_one", "local", "distance_weighted", "sparse")
SIGNAL_PROFILES = ("excitatory", "inhibitory", "mixed")
MODULATOR_PROFILES = ("none", "dopamine_like", "acetylcholine_like", "serotonin_like")
PLASTICITY_RULES = ("none", "hebbian", "oja", "stdp")
VISUALIZATION_MODES = (
    "activity",
    "potential",
    "inhibition",
    "adaptation",
    "plasticity",
    "weights",
    "spikes",
)


def _default_mechanisms() -> Dict[str, bool]:
    enabled = {
        "threshold",
        "leak",
        "refractory",
        "adaptation",
        "local_excitation",
        "lateral_inhibition",
    }
    return {name: name in enabled for name in MECHANISMS}


def _default_parameters() -> Dict[str, float]:
    return {
        "threshold": 0.60,
        "leak": 0.08,
        "refractory_ticks": 2.0,
        "adaptation_rate": 0.02,
        "excitatory_gain": 1.00,
        "inhibitory_gain": 0.70,
        "lateral_radius": 2.0,
        "recurrent_radius": 1.0,
    }


@dataclass
class LayerSpec:
    id: str
    name: str
    rows: int = 100
    cols: int = 100
    x: float = 0.50
    y: float = 0.50
    mechanisms: Dict[str, bool] = field(default_factory=_default_mechanisms)
    parameters: Dict[str, float] = field(default_factory=_default_parameters)

    @property
    def unit_count(self) -> int:
        return int(self.rows * self.cols)

    def clamp(self) -> None:
        self.rows = int(max(1, min(512, self.rows)))
        self.cols = int(max(1, min(512, self.cols)))
        self.x = float(max(0.0, min(1.0, self.x)))
        self.y = float(max(0.0, min(1.0, self.y)))

        clean = _default_mechanisms()
        for key, value in self.mechanisms.items():
            if key in clean:
                clean[key] = bool(value)
        self.mechanisms = clean

        defaults = _default_parameters()
        for key, value in self.parameters.items():
            if key in defaults:
                try:
                    defaults[key] = float(value)
                except (TypeError, ValueError):
                    pass
        self.parameters = defaults


@dataclass
class ConnectionSpec:
    id: str
    source_id: str
    target_id: str
    pattern: str = "local"
    radius: int = 3
    density: float = 0.35
    signal: str = "excitatory"
    delay: int = 1
    plasticity: str = "none"
    modulator: str = "none"
    gain: float = 1.0

    def clamp(self) -> None:
        if self.pattern not in CONNECTION_PATTERNS:
            self.pattern = "local"
        if self.signal not in SIGNAL_PROFILES:
            self.signal = "excitatory"
        if self.plasticity not in PLASTICITY_RULES:
            self.plasticity = "none"
        if self.modulator not in MODULATOR_PROFILES:
            self.modulator = "none"
        self.radius = int(max(0, min(64, self.radius)))
        self.density = float(max(0.01, min(1.0, self.density)))
        self.delay = int(max(0, min(100, self.delay)))
        self.gain = float(max(0.0, min(10.0, self.gain)))


@dataclass
class ExperimentSpec:
    name: str = "Visual Play Experiment"
    layers: List[LayerSpec] = field(default_factory=list)
    connections: List[ConnectionSpec] = field(default_factory=list)
    visualization_mode: str = "activity"
    next_layer_index: int = 1
    next_connection_index: int = 1

    @classmethod
    def default(cls) -> "ExperimentSpec":
        spec = cls()
        layer = spec.add_layer(name="Layer 1", rows=100, cols=100, x=0.52, y=0.48)
        spec.add_connection("sensor:brightness", layer.id)
        return spec

    def layer_by_id(self, layer_id: str) -> Optional[LayerSpec]:
        return next((layer for layer in self.layers if layer.id == layer_id), None)

    def connection_by_id(self, connection_id: str) -> Optional[ConnectionSpec]:
        return next((c for c in self.connections if c.id == connection_id), None)

    def source_exists(self, source_id: str) -> bool:
        return source_id in SENSORY_SOURCES or self.layer_by_id(source_id) is not None

    def add_layer(
        self,
        *,
        name: Optional[str] = None,
        rows: int = 100,
        cols: int = 100,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> LayerSpec:
        layer_id = f"layer:{self.next_layer_index}"
        display_name = name or f"Layer {self.next_layer_index}"
        if x is None:
            x = 0.36 + 0.16 * ((self.next_layer_index - 1) % 4)
        if y is None:
            y = 0.30 + 0.18 * (((self.next_layer_index - 1) // 4) % 3)
        layer = LayerSpec(layer_id, display_name, rows, cols, x, y)
        layer.clamp()
        self.layers.append(layer)
        self.next_layer_index += 1
        return layer

    def remove_layer(self, layer_id: str) -> bool:
        before = len(self.layers)
        self.layers = [layer for layer in self.layers if layer.id != layer_id]
        self.connections = [
            c for c in self.connections
            if c.source_id != layer_id and c.target_id != layer_id
        ]
        return len(self.layers) != before

    def add_connection(self, source_id: str, target_id: str) -> ConnectionSpec:
        if not self.source_exists(source_id):
            raise ValueError(f"Unknown connection source: {source_id}")
        if self.layer_by_id(target_id) is None:
            raise ValueError(f"Unknown connection target: {target_id}")
        if source_id == target_id:
            raise ValueError("Self-connections are configured as a layer mechanism.")
        if any(c.source_id == source_id and c.target_id == target_id for c in self.connections):
            raise ValueError("That connection already exists.")
        connection = ConnectionSpec(
            id=f"connection:{self.next_connection_index}",
            source_id=source_id,
            target_id=target_id,
        )
        connection.clamp()
        self.connections.append(connection)
        self.next_connection_index += 1
        return connection

    def remove_connection(self, connection_id: str) -> bool:
        before = len(self.connections)
        self.connections = [c for c in self.connections if c.id != connection_id]
        return len(self.connections) != before

    def toggle_mechanism(self, layer_id: str, mechanism: str) -> bool:
        if mechanism not in MECHANISMS:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        layer = self.layer_by_id(layer_id)
        if layer is None:
            raise ValueError(f"Unknown layer: {layer_id}")
        layer.mechanisms[mechanism] = not layer.mechanisms.get(mechanism, False)
        return layer.mechanisms[mechanism]

    def set_dimensions(
        self,
        layer_id: str,
        *,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
    ) -> LayerSpec:
        layer = self.layer_by_id(layer_id)
        if layer is None:
            raise ValueError(f"Unknown layer: {layer_id}")
        if rows is not None:
            layer.rows = int(rows)
        if cols is not None:
            layer.cols = int(cols)
        layer.clamp()
        return layer

    def set_visualization(self, mode: str) -> None:
        if mode not in VISUALIZATION_MODES:
            raise ValueError(f"Unknown visualization mode: {mode}")
        self.visualization_mode = mode

    def validate(self) -> None:
        layer_ids = [layer.id for layer in self.layers]
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("Layer IDs must be unique.")
        for layer in self.layers:
            layer.clamp()

        connection_ids = [c.id for c in self.connections]
        if len(connection_ids) != len(set(connection_ids)):
            raise ValueError("Connection IDs must be unique.")
        seen_pairs = set()
        for connection in self.connections:
            connection.clamp()
            if not self.source_exists(connection.source_id):
                raise ValueError(f"Unknown connection source: {connection.source_id}")
            if self.layer_by_id(connection.target_id) is None:
                raise ValueError(f"Unknown connection target: {connection.target_id}")
            if connection.source_id == connection.target_id:
                raise ValueError("Self-connections belong to layer mechanisms.")
            pair = (connection.source_id, connection.target_id)
            if pair in seen_pairs:
                raise ValueError(f"Duplicate connection: {pair}")
            seen_pairs.add(pair)

        if self.visualization_mode not in VISUALIZATION_MODES:
            self.visualization_mode = "activity"

        layer_nums = [
            int(layer.id.split(":", 1)[1])
            for layer in self.layers
            if layer.id.startswith("layer:") and layer.id.split(":", 1)[1].isdigit()
        ]
        conn_nums = [
            int(c.id.split(":", 1)[1])
            for c in self.connections
            if c.id.startswith("connection:") and c.id.split(":", 1)[1].isdigit()
        ]
        self.next_layer_index = max(self.next_layer_index, 1 + max(layer_nums or [0]))
        self.next_connection_index = max(self.next_connection_index, 1 + max(conn_nums or [0]))

    def to_dict(self) -> Dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ExperimentSpec":
        layers = [
            LayerSpec(**item) for item in payload.get("layers", []) if isinstance(item, dict)
        ]
        connections = [
            ConnectionSpec(**item)
            for item in payload.get("connections", [])
            if isinstance(item, dict)
        ]
        spec = cls(
            name=str(payload.get("name", "Visual Play Experiment")),
            layers=layers,
            connections=connections,
            visualization_mode=str(payload.get("visualization_mode", "activity")),
            next_layer_index=int(payload.get("next_layer_index", 1)),
            next_connection_index=int(payload.get("next_connection_index", 1)),
        )
        spec.validate()
        return spec

    def save(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> "ExperimentSpec":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Experiment file must contain a JSON object.")
        return cls.from_dict(payload)
