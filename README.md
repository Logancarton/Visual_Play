# Visual_Play

Visual_Play currently has one live early-retinal signal path:

`webcam -> 256x144 brightness -> graded retinal branches -> positive-variation spiking cascade`

The first two panels remain the physical webcam image and measured luminance. The lower panels show seven functional branches away from the picture itself:

- positive luminance variation relative to each location's adapting temporal baseline
- negative luminance variation relative to each location's adapting temporal baseline
- local contrast from each location compared with its immediate spatial surround
- left-to-right directional flow from opponent timing across adjacent locations
- right-to-left directional flow from the exact mirrored opponent timing
- top-to-bottom directional flow from opponent timing across adjacent locations
- bottom-to-top directional flow from the exact mirrored opponent timing

One cardinal directional-flow owner maintains a single shared positive trace and negative trace. The horizontal and vertical axis owners consume that same delayed history and each perform one signed opponent calculation per adjacent pair. Left-to-right evidence drives the rightward field while the opposite sign drives the leftward field. Top-to-bottom evidence drives the downward field while the opposite sign drives the upward field. Both bright and dark moving changes can drive direction. This is not OpenCV optical flow and not an object tracker.

The seven retinal branches remain graded and unchanged: 258,048 graded branch neurons at the live 256x144 resolution. Positive variation now also supplies held external current to a reusable two-field cascade containing 73,728 discrete spiking neurons. Each spiking neuron has vectorized membrane, adaptation, sustaining-current, last-spike, and refractory state. Membrane integration, threshold crossing, timestamped spikes, signed synaptic weights, explicit delays, and downstream arrival are live.

The cascade advances on its own 1 ms neural clock between camera timestamps, so its 8 ms one-to-one synaptic delay is not equivalent to “next camera frame.” Adaptation and sustaining-current arrays are present but remain zero and have no active update rule yet. Recurrence and plasticity are also not live.

## Files

- `vision_input.py` — measured webcam luminance extraction only.
- `neural_field.py` — graded fields, vectorized spiking fields, delayed synapses, reusable cascade, and live retinal pathway.
- `UI.py` — diagnostic display of webcam, brightness, variation, contrast, and four cardinal flow fields.
- `real_functional_logic.md` — source of truth for live signal logic.
- `tests/test_neural_field.py` — behavioral tests for graded retinal and discrete spiking mechanisms.

## Run

```bash
python3 -m unittest discover -s tests -v
python3 UI.py
```
