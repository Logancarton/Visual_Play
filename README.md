# Visual_Play

Visual_Play currently has one live early-retinal signal path:

`webcam -> 256x144 brightness -> temporal positive/negative variation + local spatial contrast`

The first two panels remain the physical webcam image and measured luminance. The lower row shows three functional branches away from the picture itself:

- positive luminance variation relative to each location's adapting temporal baseline
- negative luminance variation relative to each location's adapting temporal baseline
- local contrast from each location compared with its immediate spatial surround

`ON`/`OFF` variation is graded, not binary firing. Local contrast is also graded. No spike model or plasticity is live yet.

## Files

- `vision_input.py` — measured webcam luminance extraction only.
- `neural_field.py` — spatial neuron substrate, explicit synapse substrate, and live retinal signal branches.
- `UI.py` — diagnostic display of webcam, brightness, positive variation, negative variation, and local contrast.
- `real_functional_logic.md` — source of truth for live signal logic and future mechanisms.
- `tests/test_neural_field.py` — behavioral tests for spatial state, synapses, and the live retinal branches.

## Run

```bash
python3 -m unittest discover -s tests -v
python3 UI.py
```
