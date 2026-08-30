# Visual_Play

Visual_Play currently has one live early-retinal signal path:

`webcam -> 256x144 brightness -> temporal variation + local contrast + first one-direction flow`

The first two panels remain the physical webcam image and measured luminance. The lower panels show four functional branches away from the picture itself:

- positive luminance variation relative to each location's adapting temporal baseline
- negative luminance variation relative to each location's adapting temporal baseline
- local contrast from each location compared with its immediate spatial surround
- left-to-right directional flow from opponent timing across adjacent locations

The directional branch is intentionally only one preferred direction for now. It uses delayed positive/negative variation, not OpenCV optical flow and not a hard-coded object tracker.

All live outputs are graded. No spike model or long-term plasticity is live yet.

## Files

- `vision_input.py` — measured webcam luminance extraction only.
- `neural_field.py` — spatial neuron substrate, explicit synapse substrate, and live retinal signal branches.
- `UI.py` — diagnostic display of webcam, brightness, variation, contrast, and rightward flow.
- `real_functional_logic.md` — source of truth for live signal logic.
- `tests/test_neural_field.py` — behavioral tests for the live spatial and temporal mechanisms.

## Run

```bash
python3 -m unittest discover -s tests -v
python3 UI.py
```
