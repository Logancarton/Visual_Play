# Visual_Play — Real Functional Logic

## Purpose

This file is the source of truth for mechanisms that are allowed to change signal flow in Visual_Play.

Add a mechanism only when it has a real state variable or explicit signal equation, a clear owner, a place in the live path, and a behavioral test proving its effect.

The same math should normally operate at every valid location within one field. Different responses should come from different input, state, timing, or connections rather than hidden special cases.

---

# 1. CURRENT LIVE PATH — IMPLEMENTED

```text
WEBCAM
  ↓
256 × 144 normalized brightness
36,864 spatial sensory locations
  ↓
  ├─ TEMPORAL VARIATION
  │      ├─ POSITIVE VARIATION    36,864 neurons
  │      └─ NEGATIVE VARIATION    36,864 neurons
  │
  ├─ LOCAL SPATIAL CONTRAST       36,864 neurons
  │
  └─ HORIZONTAL DIRECTIONAL FLOW
         ├─ LEFTWARD FLOW         36,864 neurons
         └─ RIGHTWARD FLOW        36,864 neurons
```

The live system therefore has 36,864 sensory locations and 184,320 graded branch neurons.

None of these branches are discrete spikes.

---

# 2. TEMPORAL VARIATION

Every visual location uses the same temporal rule:

```text
delta_i(t) = luminance_i(t) - baseline_i(t)

positive_i(t) = clip(gain * max( delta_i(t), 0), 0, 1)
negative_i(t) = clip(gain * max(-delta_i(t), 0), 0, 1)
```

The local baseline moves toward current luminance using elapsed time:

```text
alpha = 1 - exp(-dt / tau)

baseline_i(t+1)
    = baseline_i(t)
    + alpha * (luminance_i(t) - baseline_i(t))
```

Current constants:

```text
baseline time constant = 350 ms
variation gain         = 3.0
```

The first observed frame seeds the temporal baseline and produces zero temporal variation.

---

# 3. LOCAL SPATIAL CONTRAST

For every location, the same center-versus-surround rule is used:

```text
surround_i(t) = mean(luminance of the 8 immediate neighboring positions)

contrast_i(t)
    = clip(contrast_gain * |luminance_i(t) - surround_i(t)|, 0, 1)
```

Current constant:

```text
contrast gain = 4.0
```

Uniform regions produce approximately zero contrast. Boundaries produce stronger contrast without an explicit orientation detector.

---

# 4. HORIZONTAL DIRECTIONAL FLOW — LEFT AND RIGHT

This mechanism combines spatial relationship with temporal order.

It does not compare raw pictures and it does not use optical flow. It consumes the already-live positive and negative temporal-variation branches.

Each adjacent horizontal pair uses one shared opponent calculation.

For a pair consisting of a left location `L` and right location `R`:

```text
right_evidence =
      delayed_positive[L] * current_positive[R]
    + delayed_negative[L] * current_negative[R]

left_evidence =
      delayed_positive[R] * current_positive[L]
    + delayed_negative[R] * current_negative[L]

pair_signal = flow_gain * (right_evidence - left_evidence)

rightward[R] = clip( pair_signal, 0, 1)
leftward[L]  = clip(-pair_signal, 0, 1)
```

Interpretation:

```text
change on left
    ↓ short delay
same-polarity change on right
    ↓
RIGHTWARD FLOW

change on right
    ↓ short delay
same-polarity change on left
    ↓
LEFTWARD FLOW
```

The two directions are true opponents. Evidence for one direction suppresses the other because they are generated from the same signed pair signal rather than from two unrelated detectors.

A simultaneous change at both locations tends to cancel rather than being mislabeled as motion.

Both bright and dark moving changes can drive either direction because positive and negative variation are correlated separately before being combined.

The delayed state is one shared low-pass temporal trace:

```text
trace(t+1) = trace(t) + alpha * (current_activity - trace(t))

alpha = 1 - exp(-dt / flow_trace_tau)
```

Current constants:

```text
flow trace time constant = 100 ms
flow gain                = 4.0
spatial offset           = 1 horizontal cell
preferred directions     = left <-> right
```

This is a deliberately minimal opponent temporal-correlation detector. It is not claimed to be a complete biological retinal direction-selective circuit.

---

# 5. LIVE OWNERSHIP

`RetinalSignalPathway` sequences the live branches.

It owns:

```text
per-location temporal baseline
last update timestamp
positive-variation field
negative-variation field
local-contrast field
horizontal directional-flow owner
```

`HorizontalDirectionalFlowField` owns:

```text
shared positive temporal trace
shared negative temporal trace
leftward output field
rightward output field
flow trace timing
```

The UI only observes these arrays.

---

# 6. NOT YET LIVE

```text
upward flow
downward flow
diagonal directions
discrete firing
per-fire timestamps
refractory behavior
orientation-specific fields
long-term plasticity
homeostatic regulation
structural growth / pruning
```

---

# 7. CURRENT TEST GATE

The live path must prove:

```text
left change then right change
   ↓
rightward response, leftward quiet

right change then left change
   ↓
leftward response, rightward quiet

simultaneous adjacent change
   ↓
both directions cancel

dark horizontal motion
   ↓
correct direction response

live brightness sequence
   ↓
variation branches
   ↓
correct horizontal direction field

malformed input or backward time
   ↓
fail without corrupting prior state
```
