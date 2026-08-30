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
  ├─ HORIZONTAL DIRECTIONAL FLOW
  │      ├─ LEFTWARD FLOW         36,864 neurons
  │      └─ RIGHTWARD FLOW        36,864 neurons
  │
  └─ VERTICAL DIRECTIONAL FLOW
         ├─ UPWARD FLOW           36,864 neurons
         └─ DOWNWARD FLOW         36,864 neurons

POSITIVE VARIATION (graded external current)
  ↓
SPIKING FIELD A                   36,864 neurons
  ↓ timestamped spikes
ONE-TO-ONE SYNAPSE PROJECTION     signed weight + 8 ms delay
  ↓ ring-buffered arrival
SPIKING FIELD B                   36,864 neurons
```

The live system therefore has 36,864 sensory locations, 258,048 graded retinal branch neurons, and 73,728 discrete spiking neurons.

The retinal branches remain graded. Discrete firing begins only in the positive-variation cascade downstream of the retina.

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

Horizontal and vertical comparisons consume the same shared low-pass temporal trace. The cardinal flow owner advances that trace once, after every axis has compared the delayed state with current activity:

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

# 5. VERTICAL DIRECTIONAL FLOW — UP AND DOWN

Vertical flow uses the same timing rule as horizontal flow, but only across adjacent rows. It consumes the positive and negative temporal-variation branches; it does not compare raw pictures.

For a pair consisting of a top location `T` and bottom location `B` in the same column:

```text
down_evidence =
      delayed_positive[T] * current_positive[B]
    + delayed_negative[T] * current_negative[B]

up_evidence =
      delayed_positive[B] * current_positive[T]
    + delayed_negative[B] * current_negative[T]

pair_signal = flow_gain * (down_evidence - up_evidence)

downward[B] = clip( pair_signal, 0, 1)
upward[T]   = clip(-pair_signal, 0, 1)
```

The sign convention is therefore explicit: positive signed evidence means top-to-bottom motion and drives the downward field; negative signed evidence means bottom-to-top motion and drives the upward field.

The vertical comparison consumes the same positive temporal trace and negative temporal trace as the horizontal comparison. Simultaneous change at both members of a pair cancels in the opponent subtraction. Bright and dark moving changes are correlated separately, so either polarity can drive either direction.

Vertical flow uses the same trace update and constants as horizontal flow:

```text
flow trace time constant = 100 ms
flow gain                = 4.0
spatial offset           = 1 vertical cell
preferred directions     = up <-> down
```

This is the vertical counterpart of the same minimal opponent temporal-correlation mechanism, not a complete optical-flow system.

---

# 6. REUSABLE SPIKING NEURAL CASCADE

The positive-variation field remains a graded retinal output. It is also the first live external current source for a general spiking substrate:

```text
positive variation
  ↓ held external current
SpikingNeuronField A
  ↓ threshold-crossing SpikeEventBatch
SynapseProjection
  ↓ signed weight after explicit delay
ring-buffer arrival
  ↓
SpikingNeuronField B
```

`SpikingNeuronField` stores one contiguous value per stable neuron ID:

```text
V[i]                         membrane potential
A[i]                         adaptation state
H[i]                         sustaining-current state
last_spike_ms[i]             most recent spike time
refractory_until_ms[i]       earliest next eligible spike time
```

The live membrane step is:

```text
V += dt / membrane_tau * (
      -V
      + input_gain * external_drive
      + H
      - A
)

V += arriving_synaptic_input
```

An eligible neuron whose membrane reaches threshold emits a real event containing its stable neuron ID and neural-clock timestamp. It then records that timestamp, sets its refractory deadline, and resets its membrane potential. During the refractory interval its membrane remains at reset and it cannot emit another spike.

`A` and `H` are real per-neuron state arrays, but both are initialized to zero and have no active evolution rule in this experiment. Adaptation and sustained-activity behavior are therefore represented but not implemented.

`SynapseProjection` remains the only connectivity authority. Every sparse connection owns:

```text
source neuron ID
target neuron ID
signed weight
delay in milliseconds
```

Its pre-existing graded `propagate()` behavior remains available and ignores delay, because it computes an instantaneous weighted projection. `NeuronSignalCascade` uses the same projection's delays for discrete spikes.

The cascade holds the last external drive and advances membrane, refractory, spike, and arrival state with a 1 ms internal neural timestep between camera timestamps. Delays are rounded upward to neural ticks, so an arrival is never delivered earlier than its configured delay. A vectorized circular buffer accumulates target input by delivery slot; there is no Python event object or heap entry per synapse event.

Current live positive-variation experiment:

```text
neural timestep           = 1 ms
membrane time constant    = 10 ms
threshold                 = 1.0
reset potential           = 0.0
refractory interval       = 5 ms
external input gain       = 4.0
projection                = one-to-one
synaptic weight           = +0.7
synaptic delay            = 8 ms
```

This is a real feed-forward temporal cascade and a reusable neural-substrate layer. It is not yet recurrent, plastic, adaptive, or a complete artificial nervous system.

---

# 7. LIVE OWNERSHIP

`RetinalSignalPathway` sequences the live branches.

It owns:

```text
per-location temporal baseline
last update timestamp
positive-variation field
negative-variation field
local-contrast field
cardinal directional-flow owner
positive-variation neuron cascade owner
```

`CardinalDirectionalFlowField` sequences the directional comparisons and owns:

```text
one shared positive temporal trace
one shared negative temporal trace
flow trace timing and once-per-frame trace advancement
horizontal directional-flow owner
vertical directional-flow owner
```

`HorizontalDirectionalFlowField` owns:

```text
leftward output field
rightward output field
horizontal adjacent-pair opponent calculation
```

`VerticalDirectionalFlowField` owns:

```text
upward output field
downward output field
vertical adjacent-pair opponent calculation
```

The axis owners receive delayed activity from the cardinal owner; neither stores a private temporal-history copy. The UI only observes the resulting arrays.

`NeuronSignalCascade` owns:

```text
upstream SpikingNeuronField
downstream SpikingNeuronField
SynapseProjection
held external drive
internal neural clock
delayed-arrival circular buffer
```

The UI does not sequence, calculate, or display the spiking cascade in a new panel during this stage.

---

# 8. NOT YET LIVE

```text
diagonal directions
orientation-specific fields
adaptation-state evolution
sustaining-current evolution
recurrent spiking connections
variable firing thresholds
synaptic plasticity / STDP
long-term plasticity
homeostatic regulation
structural growth / pruning
```

---

# 9. CURRENT TEST GATE

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

top change then bottom change
   ↓
downward response, upward quiet

bottom change then top change
   ↓
upward response, downward quiet

simultaneous adjacent vertical change
   ↓
both vertical directions cancel

dark vertical motion
   ↓
correct vertical direction response

horizontal and vertical comparison owners
   ↓
consume one shared positive/negative temporal trace

live brightness sequence
   ↓
variation branches
   ↓
correct horizontal or vertical direction field

malformed input or backward time
   ↓
fail without corrupting prior state

repeated subthreshold current
   ↓
membrane accumulation without premature firing
   ↓
threshold crossing produces neuron ID + timestamp

spike followed by strong refractory input
   ↓
no immediate refiring

configured signed and delayed synapses
   ↓
only connected targets change
   ↓
no arrival before delay
   ↓
excitatory and inhibitory membrane effects at the correct neural time

synthetic brightness increase
   ↓
graded positive variation
   ↓
upstream membrane accumulation and spike
   ↓ 8 ms
downstream membrane change
```
