# Visual_Play

Visual_Play currently has one live early-retinal signal path:

`webcam -> 256x144 brightness -> temporal variation + local contrast + paired horizontal and vertical directional flow`

The first two panels remain the physical webcam image and measured luminance. The lower panels show seven functional branches away from the picture itself:

- positive luminance variation relative to each location's adapting temporal baseline
- negative luminance variation relative to each location's adapting temporal baseline
- local contrast from each location compared with its immediate spatial surround
- left-to-right directional flow from opponent timing across adjacent locations
- right-to-left directional flow from the exact mirrored opponent timing
- top-to-bottom directional flow from opponent timing across adjacent locations
- bottom-to-top directional flow from the exact mirrored opponent timing

One cardinal directional-flow owner maintains a single shared positive trace and negative trace. The horizontal and vertical axis owners consume that same delayed history and each perform one signed opponent calculation per adjacent pair. Left-to-right evidence drives the rightward field while the opposite sign drives the leftward field. Top-to-bottom evidence drives the downward field while the opposite sign drives the upward field. Both bright and dark moving changes can drive direction. This is not OpenCV optical flow and not an object tracker.

All live outputs are graded. No spike model or long-term plasticity is live yet.

## Files

- `vision_input.py` — measured webcam luminance extraction only.
- `neural_field.py` — spatial neuron substrate, explicit synapse substrate, and live retinal signal branches.
- `UI.py` — diagnostic display of webcam, brightness, variation, contrast, and four cardinal flow fields.
- `real_functional_logic.md` — source of truth for live signal logic.
- `tests/test_neural_field.py` — behavioral tests for the live spatial and temporal mechanisms.

## Run

```bash
python3 -m unittest discover -s tests -v
python3 UI.py
```
