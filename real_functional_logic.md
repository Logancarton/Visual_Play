# Visual_Play — Real Functional Logic

## Purpose

This file is the source of truth for mechanisms that are allowed to change signal flow in Visual_Play.

Add a mechanism only when it has a real state variable, an explicit update rule, a clear owner, a place in the live signal path, and a behavioral test proving its effect.

The goal is not to attach neuroscience labels to code. The goal is to build a small spatial nervous system whose behavior changes for understandable mathematical reasons.

## Core rule

A neuron has a stable identity and spatial location. The same neuron math should normally operate everywhere within a field. Different behavior should emerge from different input, state, timing, and connections rather than from hidden special-case code.

Plasticity, when added later, should normally change influence or connectivity rather than physically moving a retinotopic neuron.

## Refined mechanism definitions

1. **Membrane / activation state** — The neuron's current internal signal level. It integrates incoming influence over time and determines how close the neuron is to responding.
2. **Excitability / threshold** — How much input the neuron needs before it meaningfully activates. Lower threshold means the neuron is easier to recruit.
3. **Adaptation** — A neuron's response gradually weakens when the same stimulation continues, making persistent signals less dominant.
4. **Refractory state** — A temporary reduction in responsiveness immediately after strong activation, preventing uninterrupted repeated firing or activation.
5. **Synaptic influence** — The amount one neuron can affect another through a specific connection.
6. **Excitation** — Synaptic influence that increases the receiving neuron's likelihood or strength of activation.
7. **Inhibition** — Synaptic influence that suppresses or counteracts activation in the receiving neuron.
8. **Synaptic strength** — The persistent weight of a connection; it determines how much of the source neuron's signal reaches its target.
9. **Transmission timing / delay** — How long a signal takes to influence another neuron, allowing temporal relationships and sequence to matter.
10. **Short-term synaptic plasticity** — Temporary changes in connection effectiveness caused by recent use, followed by recovery toward baseline.
11. **Long-term plasticity** — Experience-dependent changes that persist and alter how future signals travel through the network.
12. **Homeostasis** — Slow self-regulation that prevents neurons or networks from becoming permanently overactive or silent.
13. **Structural growth** — Creating new synaptic relationships when repeated activity patterns support a useful connection.
14. **Pruning** — Weakening and eventually removing connections that provide little useful influence.
15. **Modulatory gating** — A separate signal changes whether transmission or learning should be amplified, suppressed, or permitted at that moment.

```text
signal state = what is happening now
plasticity   = what changes because it happened
homeostasis  = what keeps those changes stable
```

---

# 1. CURRENT LIVE PATH — IMPLEMENTED

```text
WEBCAM
  ↓
256 × 144 normalized brightness
36,864 spatial sensory locations
  ↓
LOCAL ADAPTING BASELINE AT EACH (x,y)
  ↓
Δ = current brightness - local baseline
  ↓
        ┌─────────────────────┐
        │                     │
        ▼                     ▼
POSITIVE VARIATION       NEGATIVE VARIATION
ON-like branch           OFF-like branch
36,864 neurons           36,864 neurons
```

This is the first live branch away from representing the input as a single picture.

`ON` and `OFF` do **not** mean binary firing here. They mean the magnitude and direction of luminance change relative to each location's own adapting baseline.

## Per-location math

Every visual location uses the same rule.

For location `i`:

```text
Δ_i(t) = luminance_i(t) - baseline_i(t)

ON_i(t)  = clip(gain × max( Δ_i(t), 0), 0, 1)
OFF_i(t) = clip(gain × max(-Δ_i(t), 0), 0, 1)
```

The local baseline then moves toward the current luminance using elapsed time:

```text
alpha = 1 - exp(-dt / tau)

baseline_i(t+1)
    = baseline_i(t)
    + alpha × (luminance_i(t) - baseline_i(t))
```

Current live constants:

```text
baseline time constant tau = 350 ms
variation gain             = 3.0
```

The first observed frame seeds the baseline and produces zero variation output. This prevents Visual_Play from pretending the entire initial picture is a visual change.

## What this means

A stable region fades toward zero response as its local baseline catches up.

A region becoming brighter produces graded activity in the positive-change branch.

A region becoming darker produces graded activity in the negative-change branch.

Different locations do not share a global brightness baseline. Each location adapts independently while using identical math.

## Live state ownership

`RetinalVariationPathway` owns:

```text
baseline[x,y]
last update timestamp
positive-change field
negative-change field
```

Each branch field has stable spatial neuron IDs and XY coordinates through `SpatialNeuronField`.

There is no firing, refractory behavior, local inhibition, receptive-field mixing, or plasticity in this live retinal split yet.

---

# 2. EXISTING SUBSTRATE — BUILT, NOT CURRENTLY THE LIVE RETINAL ROUTE

`SpatialNeuronField` still provides:

```text
stable neuron ID
stable x,y position
potential
activity
threshold
leak
maximum potential
```

Its current graded state equation is:

```text
V_j(t+1) = clip((1 - leak) × V_j(t) + drive_j(t), 0, Vmax)

A_j(t+1) = clip((V_j(t+1) - threshold) / (1 - threshold), 0, 1)
```

`SynapseProjection` still provides explicit source-neuron, target-neuron, and weight arrays with pairwise propagation:

```text
synaptic_drive_j = Σ_i activity_i × weight_ij
```

These are valid substrate mechanisms, but the current lower two UI panels no longer show the old one-to-one Depth-0 -> Depth-1 relay.

---

# 3. NEXT SIGNAL MECHANISMS — NOT YET LIVE

Potential future mechanisms include:

```text
membrane integration
thresholded firing
per-fire timestamps
refractory state
local excitation
lateral inhibition
center-surround receptive fields
transmission delay
short-term synaptic state
long-term plasticity
homeostatic regulation
structural growth / pruning
modulatory gating
```

They should be added one at a time and only when the current signal path gives them a real job.

## Firing rule requirement

When firing is added, every firing event should have simulation time attached to it. At minimum:

```text
neuron_id
timestamp_ms
previous_fire_ms
fire_count
```

Do not call graded ON/OFF variation a spike.

## STDP requirement

Do not implement STDP until real discrete firing events and timing traces exist. A graded coactivity equation is not STDP.

---

# 4. TEST GATE FOR EACH NEW MECHANISM

A mechanism is not complete because an equation exists.

It should prove:

1. the live camera path reaches it
2. valid input changes only state it owns
3. malformed input fails without corrupting prior state
4. spatial identity remains correct
5. its downstream effect has the expected direction
6. time-dependent behavior uses elapsed time rather than frame count when appropriate
7. reset clears the correct state
8. values remain finite and bounded
9. the UI reports underlying state rather than fabricating an image

For the current retinal split specifically:

```text
first frame
   ↓
seed local baseline
   ↓
no fabricated variation

brighter at (x,y)
   ↓
positive branch at same (x,y)

 darker at (x,y)
   ↓
negative branch at same (x,y)

unchanged input
   ↓
baseline adapts
   ↓
variation response fades
```
