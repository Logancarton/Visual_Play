# Visual_Play — Real Functional Logic

## Purpose

This file is the source of truth for mechanisms that are allowed to change signal flow in Visual_Play.

The project should add mechanisms only when they have a clear causal owner, a real state variable, an explicit update rule, a place in the live signal path, and a behavioral test that proves their effect.

The goal is not to label code with neuroscience terms. The goal is to build a small computational nervous system in which signal flow changes for understandable reasons.

## Core rule

A neuron is not a UI object and a field is not a moving box.

A neuron has a stable identity and spatial location. Functional change should normally occur through:

- membrane / activation state
- excitability and threshold
- adaptation and refractory state
- incoming and outgoing synaptic influence
- excitation and inhibition
- synaptic strength
- transmission timing
- short-term synaptic state
- long-term plasticity
- homeostatic regulation
- structural growth or pruning
- modulatory gating

Plasticity should normally change connectivity or influence, not physically move a retinotopic neuron. If a receptive field appears to “move,” that should come from changing synaptic influence while the neuron itself retains its identity.

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

The central distinction is:

```text
signal state = what is happening now
plasticity   = what changes because it happened
homeostasis  = what keeps those changes stable
```

---

## 1. Current live pathway — IMPLEMENTED

```text
webcam
   ↓
64 × 36 normalized brightness
   ↓
Depth 0 spatial neuron field
2,304 neurons
   ↓
2,304 explicit one-to-one synapses
weight = 0.85
   ↓
Depth 1 spatial neuron field
2,304 neurons
```

### Current neuron state

Every `SpatialNeuronField` currently owns:

```text
neuron_id
x, y position
potential
activity
threshold
leak
maximum potential
```

Neuron identity and XY position are stable. Potential and activity change every tick.

### Current neuron equation

For neuron `j`:

```text
V_j(t+1) = clip((1 - leak_j) × V_j(t) + D_j(t), 0, Vmax)

A_j(t+1) = clip((V_j(t+1) - threshold_j) / (1 - threshold_j), 0, 1)
```

Where:

- `V` = membrane-like potential state
- `D` = total incoming drive
- `A` = graded output activity

This is a useful stateful neuron substrate. It is not yet a biological spiking neuron model.

### Current synapse equation

Each synapse stores an explicit source neuron, target neuron, and weight.

For target neuron `j`:

```text
synaptic_drive_j(t) = Σ_i A_i(t) × w_ij
```

The current live pathway uses one synapse from each Depth 0 neuron to the neuron with the same XY identity in Depth 1.

That means retinotopy is currently specified by the initial wiring. It has not been learned.

---

## 2. Functional signal model — TARGET ARCHITECTURE

A useful general signal equation is:

```text
external sensory drive
        +
excitatory synaptic drive
        -
inhibitory synaptic drive
        +
local / recurrent influence
        -
adaptation
        +
modulatory gain
        ↓
membrane state
        ↓
threshold / excitability
        ↓
activity or spike event
        ↓
synapses
        ↓
next neurons
```

A future tick should conceptually become:

```text
I_j(t) = E_j(t)
       + Σ excitation_ij(t)
       - Σ inhibition_ij(t)
       + R_j(t)
       - Adapt_j(t)

V_j(t+1) = membrane_update(V_j(t), I_j(t), leak_j, refractory_j)

A_j(t+1) = activation(V_j(t+1), threshold_j)
```

Each mechanism below modifies one of these causal terms. No mechanism should bypass this flow by directly painting an output image or mutating unrelated state.

---

## 3. Ownership boundaries

### Neuron owner

A neuron may own:

```text
stable neuron ID
stable spatial coordinate
potential
activity
threshold / excitability
refractory state
adaptation state
running activity average
intrinsic-plasticity state
```

### Synapse owner

A synapse may own:

```text
source neuron ID
target neuron ID
weight
excitatory / inhibitory sign
transmission delay
short-term facilitation state
short-term depression state
eligibility / learning trace
long-term usage statistics
```

### Field owner

A field may own rules that depend on spatial relationships among neurons:

```text
local-neighbor topology
lateral inhibition
local recurrent excitation
activity normalization
homeostatic targets
```

A field should not hide individual synapses behind a single whole-field weight.

### Pathway / runtime owner

The pathway should remain thin. It may:

```text
accept sensory drive
sequence field updates
route synaptic drive
invoke plasticity after pre/post activity is known
return observable state
```

It should not become the owner of learning equations that belong to neurons or synapses.

### UI owner

The UI observes state only. It should not create neural behavior.

---

# MECHANISMS THAT CHANGE SIGNAL FLOW

## 4. Synaptic weight — immediate influence + long-term memory

A synaptic weight determines how strongly a source neuron influences a target.

```text
contribution_ij = source_activity_i × weight_ij
```

Large positive weight → stronger excitation.

Negative or separately typed inhibitory connection → stronger inhibition.

Weight is the first major location for learned long-term change.

Status: **explicit weights exist; learning does not yet exist.**

---

## 5. Excitation and inhibition

The nervous system does not only forward positive activity. Some signals increase the probability of downstream activity while others suppress it.

Target drive should eventually separate these terms:

```text
net_synaptic_drive_j = excitatory_drive_j - inhibitory_drive_j
```

Important uses:

- local competition
- contrast enhancement
- edge / boundary sensitivity
- suppressing redundant activity
- stabilizing recurrent networks
- winner-takes-some behavior

The current `SynapseProjection` can mathematically carry negative weights, but Visual_Play does not yet have a deliberate live inhibitory circuit.

Status: **substrate-capable, not live as an organized mechanism.**

---

## 6. Leak

Leak causes old potential to fade when input stops.

```text
retained_potential = (1 - leak) × previous_potential
```

Higher leak:

```text
faster forgetting of momentary electrical state
less temporal accumulation
more dependence on current input
```

Lower leak:

```text
longer temporal integration
more persistence
more risk of saturation
```

Status: **IMPLEMENTED.**

---

## 7. Threshold / intrinsic excitability

Threshold determines how much potential is required before a neuron becomes active.

This is different from synaptic plasticity. Two neurons can receive the same drive but respond differently because their intrinsic excitability differs.

Future intrinsic plasticity may alter threshold slowly:

```text
high sustained activity → threshold rises
low sustained activity  → threshold falls
```

This creates another form of memory: the neuron itself changes how easily it responds.

Status: **fixed threshold implemented; adaptive threshold not implemented.**

---

## 8. Refractory behavior

After a strong activation or future spike, a neuron should not necessarily be able to immediately produce the same response again.

A refractory state can temporarily:

```text
raise threshold
reduce gain
or prevent another spike/event
```

This affects timing, prevents uncontrolled rapid reactivation, and becomes essential if Visual_Play moves to event/spike-based neurons.

Status: **NOT IMPLEMENTED.**

---

## 9. Adaptation / activity fatigue

A neuron that remains continuously active should often become less responsive to an unchanged stimulus.

One simple state variable is:

```text
Adapt_j(t+1) = decay × Adapt_j(t) + rate × A_j(t)

net_drive_j = incoming_drive_j - Adapt_j
```

Effect:

```text
new / changing signal → relatively strong
unchanging signal     → progressively reduced
```

This can create novelty sensitivity without explicitly programming “novelty detection.”

Status: **NOT IMPLEMENTED.**

---

## 10. Local recurrent excitation

Nearby neurons may excite nearby neurons.

```text
neuron j
  ↑  ↖  ↗
nearby active neurons
```

This can support:

- continuation of coherent spatial patterns
- local grouping
- persistence
- amplification of weak but spatially consistent signals

It must be balanced by inhibition/homeostasis or it can create runaway activity.

Status: **NOT IMPLEMENTED.**

---

## 11. Lateral inhibition

Nearby activity can suppress surrounding activity.

Conceptually:

```text
        inhibit
     ○ ○ ○ ○ ○
     ○ ○ ● ○ ○
     ○ ● + ● ○
     ○ ○ ● ○ ○
     ○ ○ ○ ○ ○
```

A neuron can therefore respond not only to absolute brightness but to brightness relative to its neighborhood.

This is a major candidate for producing contrast and boundary structure from neuronal interaction rather than from OpenCV preprocessing.

Status: **NOT IMPLEMENTED.**

---

## 12. Transmission delay

A synapse does not have to affect its target in the same tick.

```text
source event at t
       ↓
connection delay d
       ↓
target influence at t+d
```

Delay creates temporal sequence information and becomes important for motion, direction sensitivity, synchronization, oscillation, and STDP.

Status: **NOT IMPLEMENTED.**

---

# NEUROPLASTICITY

## 13. Long-term synaptic plasticity

Long-term plasticity changes `w_ij` based on experience.

The central rule is:

```text
experience changes future signal flow
```

A repeated pattern should therefore change which routes become easier or harder to activate later.

### Current-model-compatible first rule: Oja-style graded plasticity

Because Visual_Play currently has continuous graded activity rather than discrete spike times, an Oja-style rule is a cleaner first learning mechanism than pretending STDP exists.

For source activity `x_i`, target activity `y_j`, and weight `w_ij`:

```text
Δw_ij = η × y_j × (x_i - y_j × w_ij)

w_ij ← clip(w_ij + Δw_ij, w_min, w_max)
```

Why this is useful:

- correlated pre/post activity strengthens influence
- the normalization term resists unlimited weight growth
- it works directly with the graded activity already produced by the current neuron model
- learning remains local to information available at the synapse

This is a computational learning rule inspired by Hebbian/Oja learning. It should not be described as a complete biological LTP/LTD model.

Status: **NEXT CANDIDATE.**

### Hebbian alternative

The simplest rule is:

```text
Δw_ij = η × pre_i × post_j
```

But pure Hebbian learning is unstable because weights can grow indefinitely. It therefore requires weight decay, normalization, competition, or homeostasis.

For Visual_Play, pure Hebbian learning should not be added alone.

---

## 14. STDP — blocked until real timing exists

Spike-timing-dependent plasticity depends on the relative timing of discrete pre- and post-synaptic events.

Conceptually:

```text
pre before post → strengthen
post before pre → weaken
```

Visual_Play currently has graded activity values and no true spike-event timestamps. Calling a graded coactivity equation “STDP” would be false.

STDP should only be implemented after neurons generate discrete events/spikes and synapses can retain timing traces.

Status: **BLOCKED BY SPIKE/EVENT MODEL.**

---

## 15. Short-term synaptic plasticity

Synaptic strength can change temporarily without becoming long-term memory.

### Facilitation

Repeated recent activity can temporarily increase effective transmission.

```text
effective_weight = long_term_weight × facilitation_state
```

### Depression

Repeated recent activity can temporarily reduce available transmission.

```text
effective_weight = long_term_weight × available_release_state
```

These mechanisms allow identical long-term synapses to respond differently depending on recent history.

Status: **NOT IMPLEMENTED.**

---

## 16. Homeostatic plasticity

Learning requires a stabilizer. Otherwise a network can become silent or saturated.

A neuron should maintain a slow running activity estimate:

```text
avg_activity_j ← slow_average(A_j)
```

Then excitability can adjust toward a target:

```text
threshold_j ← threshold_j + η_h × (avg_activity_j - target_activity)
```

Therefore:

```text
too active → threshold increases
not active enough → threshold decreases
```

An alternative or complementary mechanism is synaptic scaling, where all incoming weights are slowly scaled while their relative pattern is preserved.

Homeostasis should operate much slower than ordinary signal propagation and ordinary learning.

Status: **HIGH-PRIORITY STABILIZER; NOT IMPLEMENTED.**

---

## 17. Structural plasticity — connection growth and pruning

Long-term learning can eventually alter topology itself.

### Pruning

A connection may become removable when all of these remain true for a sufficiently long period:

```text
very weak weight
+
very low usage
+
low contribution to target activity
```

### Growth

A new connection may become eligible when neurons repeatedly show a meaningful local relationship and the target has available connection capacity.

Growth should not search the entire network indiscriminately. Candidate connections should initially be constrained by biologically/architecturally meaningful neighborhoods.

Structural plasticity should occur much more slowly than weight plasticity.

Status: **LATER. Requires stable weight learning and usage statistics first.**

---

## 18. Neuromodulation

Neuromodulatory signals should change how strongly other mechanisms operate rather than becoming another ordinary sensory channel.

A modulator might alter:

```text
learning rate
neuron gain
threshold
plasticity eligibility
adaptation rate
```

For example:

```text
Δw_ij = modulator(t) × η × local_learning_signal_ij
```

This gives Visual_Play a future mechanism for states such as salience, reward, uncertainty, attention, or behavioral relevance to gate learning.

Do not add dopamine/acetylcholine/serotonin labels until a real functional signal exists that justifies the modeled effect.

Status: **LATER.**

---

## 19. Noise / stochasticity

Biological neural systems are not perfectly deterministic. Controlled noise can eventually affect:

```text
membrane potential
release probability
threshold
exploration of weak connections
```

Noise should only be added when there is a hypothesis to test. It should never be used to make behavior merely “look biological.”

Status: **LATER / EXPERIMENTAL.**

---

# FUNCTIONAL RECEPTIVE FIELDS

## 20. Receptive field change is not neuron movement

A Depth 1 neuron may remain at the same physical/retinotopic coordinate while the pattern of Depth 0 neurons influencing it changes.

Example:

```text
before learning

. . .
. ● .  → target neuron
. . .

later

. + +
. ● +  → same target neuron
. . .
```

The neuron did not move. Its receptive influence changed.

A useful derived receptive-field center can be measured from incoming weights:

```text
RF_center_j = Σ_i |w_ij| × position_i / Σ_i |w_ij|
```

If that derived center changes, the UI may visualize the receptive field as shifting. That is functional remapping, not physical neuron migration.

---

# TIMESCALES

## 21. Mechanisms should operate at different speeds

```text
EVERY TICK
sensory input
synaptic transmission
leak
excitation / inhibition
thresholded activation
refractory state

FAST HISTORY
adaptation
short-term facilitation / depression
transmission traces

LEARNING
Oja / Hebbian-like weight change
intrinsic excitability change

SLOW STABILIZATION
homeostatic threshold regulation
synaptic scaling

VERY SLOW
structural growth
pruning
long-term receptive-field reorganization
```

If every mechanism changes at the same speed, the system will be unstable and biologically incoherent.

---

# IMPLEMENTATION ORDER

## Stage 0 — current substrate — DONE

```text
brightness
  ↓
stateful spatial neurons
  ↓
explicit one-to-one synapses
  ↓
stateful downstream neurons
```

Proved:

- stable neuron IDs
- stable XY positions
- potential and activity state
- explicit source-target synapses
- weighted propagation
- live webcam signal reaches the next depth through those synapses

## Stage 1 — local receptive-field substrate

Replace the downstream field's single guaranteed source with a small local candidate neighborhood.

Initial experiment:

```text
for each Depth 1 neuron
    connect from nearby Depth 0 neurons
    radius = 1 cell initially
    up to 3 × 3 = 9 candidate inputs
```

This creates roughly 20,000 synapses at the current 64 × 36 resolution, which remains inexpensive while allowing actual competition among possible routes.

Requirements:

- neuron positions remain fixed
- connections remain explicit source-target pairs
- no global dense matrix
- spatial neighborhood is based on actual XY identity
- malformed topology fails safely

## Stage 2 — first live neuroplasticity

Add graded Oja-style weight plasticity to `SynapseProjection` or the appropriate synapse owner.

Required flow:

```text
Depth 0 activity
      ↓
propagate through current weights
      ↓
Depth 1 activity
      ↓
pre + post activity become available
      ↓
update only the participating synapses
      ↓
new weights alter the next signal
```

This is the first point at which experience changes future signal flow.

Required safeguards:

- bounded weights
- learning can be enabled/disabled
- no NaN/Inf mutation
- inactive/unrelated routes do not mutate unexpectedly
- activity reset does not silently erase learned weights
- learning reset is a separate explicit operation

## Stage 3 — homeostatic stabilization

Add slow activity tracking and adaptive threshold or synaptic scaling.

Goal:

```text
plasticity can learn
without all neurons becoming permanently ON or OFF
```

## Stage 4 — local excitation and lateral inhibition

Add real within-field local circuits.

Goal:

```text
spatial relationships begin altering the representation itself
```

This is where contrast, boundary emphasis, local competition, and coherent grouping can begin emerging from neuron interactions.

## Stage 5 — adaptation / short-term synaptic state

Add temporal history mechanisms so identical current input can produce different output depending on recent experience.

## Stage 6 — event/spike timing

Only if useful, add explicit neuronal events/spikes and transmission timing.

This unlocks legitimate:

- refractory periods
- conduction-delay experiments
- temporal coding
- STDP

## Stage 7 — structural plasticity

Only after weight learning is stable:

```text
grow useful candidate routes
prune persistently unused routes
```

## Stage 8 — modulatory learning gates

Only after Visual_Play has a real source of salience/reward/task relevance.

---

# FIRST PLASTICITY EXPERIMENT

## Local receptive-field learning v1

The first experiment should answer one narrow question:

> Can repeated spatial input change the strength of real local neuron-to-neuron routes while preserving stability and spatial identity?

Proposed setup:

```text
Depth 0: 64 × 36 neurons

Each Depth 1 neuron:
    receives candidate synapses from its local 3 × 3 Depth 0 neighborhood

Initial weights:
    small, bounded, normalized local weights

Learning:
    Oja-style graded update

Stabilization:
    hard weight bounds initially
    homeostasis added immediately after the learning mechanism is proven
```

Observable evidence of learning:

```text
same repeated stimulus before learning
           ↓
weight pattern changes
           ↓
same stimulus later
           ↓
different downstream activity because the weights changed
```

That is a real functional definition of learning for Visual_Play.

The UI should eventually show:

```text
selected Depth 1 neuron
        ↓
its incoming local neurons
        ↓
current weights
        ↓
how those weights change over repeated exposure
```

The UI is evidence. The synapses own the learning.

---

# TEST GATES FOR EVERY NEW MECHANISM

A mechanism is not complete because an equation exists.

Every mechanism should prove:

1. the live webcam/neuron pathway reaches it
2. valid input changes only the state the mechanism owns
3. malformed input fails without corrupting existing state
4. unrelated neurons/synapses are not mutated
5. its effect changes downstream signal in the expected direction
6. state can be reset at the correct level
7. persistent learned state is not accidentally erased by an activity reset
8. values remain finite and bounded
9. the UI only reports the real underlying state

For plasticity specifically:

```text
pre/post relationship
      ↓
weight update
      ↓
future propagation changes
```

must be demonstrated end to end.

---
