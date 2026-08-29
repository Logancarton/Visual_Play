# Visual_Play

Visual_Play currently has one live retinal-style signal path:

`webcam -> 256x144 brightness -> per-location adapting baseline -> graded positive-change branch + graded negative-change branch`

The first two panels remain the physical webcam image and measured luminance. The two lower panels now show the first functional split away from the picture itself.

Important: `ON` and `OFF` currently mean graded positive and negative luminance variation. They are not binary spikes.

## Files

- `vision_input.py` — measured webcam luminance extraction only.
- `neural_field.py` — spatial neuron substrate, explicit synapse substrate, and the live retinal variation pathway.
- `UI.py` — diagnostic display of webcam, brightness, positive variation, and negative variation.
- `real_functional_logic.md` — source of truth for live signal logic and future mechanisms.
- `tests/test_neural_field.py` — behavioral tests for spatial state, synapses, and the retinal split.

## Run

```bash
python3 -m unittest discover -s tests -v
python3 UI.py
```
