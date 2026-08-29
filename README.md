# Visual_Play

Visual_Play is currently reduced to one live path:

`webcam -> brightness map -> spatial neuron field -> explicit synapses -> next spatial neuron field`

The project intentionally has no graph editor, cortical hierarchy, plasticity rule, or reconstruction system right now. The goal is to make the neural substrate real before adding higher cognition or richer visualization.

## Files

- `vision_input.py` — measured webcam luminance extraction only.
- `neural_field.py` — neuron identity, spatial coordinates, field state, explicit synapses, and propagation.
- `UI.py` — minimal live diagnostic display only.
- `tests/test_neural_field.py` — behavioral tests for the real signal path.

## Run

```bash
python3 -m unittest discover -s tests -v
python3 UI.py
```
