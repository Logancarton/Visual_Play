# Visual_Play — Real Functional Logic

## Purpose

This file is the source of truth for mechanisms that are allowed to change signal flow in Visual_Play.

Add a mechanism only when it has a real state variable or explicit signal equation, a clear owner, a place in the live path, and a behavioral test proving its effect.

The same math should normally operate at every location within one field. Different responses should come from different input, state, timing, or connections rather than hidden special cases.

---

# 1. CURRENT LIVE PATH — IMPLEMENTED

```text
WEBCAM
  ↓
256 × 144 normalized brightness
36,864 spatial sensory locations
  ↓
  ├──────────────── TEMPORAL COMPARISON ────────────────┐
  │                                                     │
  │ local adapting baseline at each (x,y)               │
  │ Δ = current brightness - baseline                   │
  │                                                     │
  │        ┌───────────────────┐                        │
  │        ▼                   ▼                        │
  │ POSITIVE VARIATION    NEGATIVE VARIATION            │
  │ 36,864 neurons        36,864 neurons                │
  │                                                     │
  └─────────────────────────────────────────────────────┘

  └───────────────── SPATIAL COMPARISON ────────────────┐
                                                        │
      immediate 8-neighbor surround mean                │
      |center luminance - surround luminance|           │
                        ↓                               │
                  LOCAL CONTRAST                        │
                  36,864 neurons                        │
```

The live system therefore has 36,864 sensory locations and 110,592 graded branch neurons.

None of these branches are discrete spikes.

---

# 2. TEMPORAL VARIATION

Every visual location uses the same temporal rule:

```text
Δ_i(t) = luminance_i(t) - baseline_i(t)

positive_i(t) = clip(gain × max( Δ_i(t), 0), 0, 1)
negative_i(t) = clip(gain × max(-Δ_i(t), 0), 0, 1)
```

The local baseline moves toward the current luminance using elapsed time:

```text
alpha = 1 - exp(-dt / tau)

baseline_i(t+1)
    = baseline_i(t)
    + alpha × (luminance_i(t) - baseline_i(t))
```

Current constants:

```text
baseline time constant tau = 350 ms
variation gain             = 3.0
```

The first observed frame seeds the temporal baseline and produces zero temporal variation.

---

# 3. LOCAL SPATIAL CONTRAST

Local contrast is a separate branch. It does not compare a location with its own past; it compares that location with nearby locations in the current frame.

For every location `i`, the same rule is used:

```text
surround_i(t) = mean(luminance of the 8 immediate neighboring positions)

contrast_i(t)
    = clip(contrast_gain × |luminance_i(t) - surround_i(t)|, 0, 1)
```

Current constant:

```text
contrast gain = 4.0
```

Uniform regions produce approximately zero local contrast. A bright or dark location differing from its immediate surround produces stronger contrast activity. Boundaries therefore become visible without an orientation detector being programmed.

This is a deliberately simple center-versus-surround approximation. It is not yet a full biological center-surround retinal circuit, and it does not yet distinguish horizontal, vertical, or other orientations.

Unlike temporal variation, local contrast can exist on the first frame because it depends on spatial relationships that are already present.

---

# 4. LIVE OWNERSHIP

`RetinalSignalPathway` owns:

```text
per-location temporal baseline
last update timestamp
positive-variation field
negative-variation field
local-contrast field
```

Each branch is a `SpatialNeuronField` with stable spatial neuron identity and XY coordinates.

The UI observes these arrays only. It does not manufacture contrast or variation.

`SpatialNeuronField` also provides the existing graded membrane-like substrate, and `SynapseProjection` still provides explicit source-target weighted synapses. Those synapses are built substrate but are not currently the owner of the retinal comparison math.

---

# 5. NOT YET LIVE

```text
discrete firing
per-fire timestamps
refractory behavior
motion from spatial timing
horizontal orientation
vertical orientation
other orientation channels
long-term plasticity
homeostatic regulation
structural growth / pruning
```

Do not implement STDP until real discrete firing events and timing traces exist.

---

# 6. CURRENT TEST GATE

The live retinal path must prove:

```text
first frame
   ↓
seeds temporal baseline
   ↓
no fabricated temporal variation

brighter than temporal baseline
   ↓
positive variation at same location

 darker than temporal baseline
   ↓
negative variation at same location

uniform spatial region
   ↓
near-zero local contrast

bright or dark center against surround
   ↓
local contrast at that center

malformed input or backward time
   ↓
fail without corrupting prior retinal state
```
