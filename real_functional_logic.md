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
  │      ├─ POSITIVE VARIATION   36,864 neurons
  │      └─ NEGATIVE VARIATION   36,864 neurons
  │
  ├─ LOCAL SPATIAL CONTRAST      36,864 neurons
  │
  └─ ONE DIRECTIONAL FLOW
         LEFT -> RIGHT            36,864 neurons
```

The live system therefore has 36,864 sensory locations and 147,456 graded branch neurons.

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

# 4. FIRST DIRECTIONAL FLOW — LEFT TO RIGHT

This is the first mechanism that combines spatial relationship with temporal order.

It does not compare raw pictures and it does not use optical flow. It consumes the already-live positive and negative temporal-variation branches.

Each horizontal pair uses the same opponent correlation equation.

For target location `(x,y)` with the neighbor immediately to its left:

```text
preferred =
      delayed_positive[x-1,y] * current_positive[x,y]
    + delayed_negative[x-1,y] * current_negative[x,y]

opponent =
      delayed_positive[x,y] * current_positive[x-1,y]
    + delayed_negative[x,y] * current_negative[x-1,y]

rightward[x,y]
    = clip(flow_gain * max(preferred - opponent, 0), 0, 1)
```

Interpretation:

```text
change at left
      ↓
short delay
      ↓
same-polarity change at right
      ↓
RIGHTWARD FLOW activity
```

The reverse temporal order contributes to the opponent term and suppresses the rightward response.

A simultaneous change at both locations tends to cancel rather than being mislabeled as motion.

Both bright and dark moving changes can drive the same direction field because positive and negative variation are correlated separately before being combined.

The delayed state is a low-pass temporal trace:

```text
trace(t+1) = trace(t) + alpha * (current_activity - trace(t))

alpha = 1 - exp(-dt / flow_trace_tau)
```

Current constants:

```text
flow trace time constant = 100 ms
flow gain                = 4.0
spatial offset           = 1 horizontal cell
preferred direction      = left -> right only
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
rightward directional-flow owner
```

`DirectionalFlowField` owns:

```text
positive temporal trace
negative temporal trace
rightward output field
flow trace timing
```

The UI only observes these arrays.

---

# 6. NOT YET LIVE

```text
right -> left flow
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

Do not add the other directions until this first directional detector behaves correctly in live play.

---

# 7. CURRENT TEST GATE

The live path must prove:

```text
left change then right change
   ↓
rightward response

right change then left change
   ↓
no rightward response

simultaneous adjacent change
   ↓
opponent cancellation

dark left-to-right change
   ↓
rightward response

live brightness sequence
   ↓
variation branches
   ↓
rightward field

malformed input or backward time
   ↓
fail without corrupting prior state
```
