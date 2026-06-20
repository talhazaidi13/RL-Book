# Chapter 13 Instructor Solutions

## How to use this guide

Each exercise includes four elements:

1. **Expected answer**: the core solution an instructor should look for.
2. **Teaching notes**: common misconceptions and points worth emphasizing.
3. **Grading rubric**: a suggested allocation for a 10-point question.
4. **Executable component**: where appropriate, a notebook or code module that demonstrates the idea.

The solutions are written for instructors. They are more detailed than a typical student answer and should be adapted to the level of the course.

---

# Conceptual Exercises

## Exercise 1 — Physical state versus observation

### Expected answer

The physical state \(s_t\) contains all variables needed to predict the next state under the Markov assumption. The observation \(o_t\) contains only what the robot can measure or estimate at decision time. In real robotic systems, \(o_t\) is generally incomplete, noisy, delayed, and sensor-dependent.

A manipulation example is drawer opening. The true state may include latch engagement, contact mode, friction, handle pose, and the exact gripper-object force distribution. A camera and joint encoders may reveal only the drawer pose, gripper pose, and robot configuration. Because latch engagement is hidden, a memoryless visual policy may fail. The policy may need force sensing, temporal history, or an active probing action.

A navigation example is movement through visually similar corridors. The physical state includes the robot's true global pose and map location. A single camera image may not distinguish two similar hallways. The policy therefore requires odometry, a map, a recurrent state, or a belief estimate.

The distinction changes policy design because a fully observed MDP may use \(\pi(a_t\mid s_t)\), whereas a partially observed task requires \(\pi(a_t\mid h_t)\) or \(\pi(a_t\mid b_t)\), where \(h_t\) is a history representation and \(b_t\) is a belief state.

### Teaching notes

A frequent error is to say that the observation is simply a lower-dimensional state. The main issue is not dimensionality but information availability. A high-dimensional image can still omit essential latent variables.

### Suggested rubric

- Defines state correctly: 2 points
- Defines observation correctly: 2 points
- Manipulation example: 2 points
- Navigation example: 2 points
- Connects distinction to memory, sensing, or belief-state policy: 2 points

---

## Exercise 2 — Comparison of robotic action abstractions

### Expected answer

| Action type | Main advantage | Main limitation |
|---|---|---|
| Torque command | Maximum authority; supports highly dynamic behavior | Difficult exploration, actuator-model sensitivity, and greater safety burden |
| Joint-position target | Stable with existing servo controllers; easy to enforce limits | Less expressive for dynamic contact and dependent on low-level controller quality |
| End-effector delta | Intuitive for Cartesian manipulation; reduces kinematic burden | Requires inverse kinematics and may fail near singularities or obstacles |
| Action chunk | Improves temporal consistency and can reduce policy-query frequency | Later actions can become stale; chunk length introduces feedback-latency trade-off |
| Skill token | Reduces planning horizon and supports compositional tasks | Assumes reliable skill executors and correct grounding/preconditions |

A strong answer should also state that higher-level actions reduce the search space but introduce reliance on engineered controllers or skills. Lower-level actions provide flexibility but place more responsibility on learning.

### Suggested rubric

Two points for each action type: one correct advantage and one correct limitation.

---

## Exercise 3 — Why behavior cloning is a strong baseline and why it fails

### Expected answer

Behavior cloning is a strong first baseline because it converts robot learning into supervised prediction. It is easy to optimize, requires no reward, avoids unsafe online exploration, and directly uses teleoperation or scripted demonstrations. It also provides a diagnostic test of dataset quality: if BC cannot reproduce nominal behavior, more complex methods are unlikely to solve the underlying data problem.

BC can fail in closed loop because training samples come from the demonstrator's state distribution, while deployment samples come from the learned policy's state distribution. Small errors move the robot into states that are rare or absent in the dataset. The resulting errors compound over time. This is covariate shift.

Other failure causes include multimodal action averaging, missing recovery behavior, sensor distribution shift, action-scale mismatch, and open-loop evaluation that hides compounding errors.

Corrective data collection, dataset aggregation, temporal action models, closed-loop validation, and recovery demonstrations are common remedies.

### Suggested rubric

- Why BC is attractive: 3 points
- Correct explanation of covariate shift: 4 points
- At least two additional failure modes or remedies: 3 points

---

## Exercise 4 — VLA supervised prediction versus RL optimization

### Expected answer

A supervised VLA policy minimizes an action-prediction loss such as token cross-entropy, regression error, denoising loss, or flow-matching loss. Its target is to reproduce demonstrated behavior conditioned on vision, language, and robot observations.

An RL policy maximizes expected discounted return or a constrained variant of that objective. Its update depends on consequences rather than only action labels.

The two can use the same architecture but learn different behavior. A VLA trained from demonstrations may inherit demonstrator errors and cannot automatically prefer a higher-success action that is absent from the data. RL can improve task success, recovery, efficiency, or safety, but it requires rewards or value estimates and may introduce unsafe exploration or critic error.

A VLA policy can serve as an initialization or proposal distribution, while RL can fine-tune an adapter, residual, or action head, or rank candidate action chunks.

### Suggested rubric

- Correct supervised objective: 3 points
- Correct RL objective: 3 points
- Explains behavioral consequence: 2 points
- Gives a valid VLA-RL integration example: 2 points

---

## Exercise 5 — VLA inference latency and execution horizon

### Expected answer

The low-level controller runs at \(100\) Hz, so one action interval is

\[
\Delta t = \frac{1}{100} = 0.01\ \text{s}=10\ \text{ms}.
\]

Executing \(H_{\mathrm{exec}}=8\) actions covers

\[
8\times 10\ \text{ms}=80\ \text{ms}.
\]

The VLA requires \(120\) ms to produce the next chunk. Under synchronous replanning, the current chunk ends \(40\) ms before the next chunk is available. Therefore, the action stream cannot remain continuous.

The theoretical minimum execution horizon is

\[
H_{\mathrm{exec,min}}
=
\left\lceil \frac{120\ \text{ms}}{10\ \text{ms/action}}\right\rceil
=
12.
\]

A practical implementation should use more than 12 steps because communication, preprocessing, scheduling, and safety filtering also consume time.

Increasing \(H_{\mathrm{exec}}\) reduces feedback frequency and increases the chance that later actions become invalid after contact, object movement, or perception changes.

An asynchronous alternative generates the next chunk while the robot is still executing the current one. Other alternatives include policy distillation, feature caching, fewer denoising steps, a fast low-level policy, or overlapping chunks.

### Suggested rubric

- Computes 10 ms action interval: 2 points
- Computes 80 ms coverage: 2 points
- Identifies 40 ms gap: 2 points
- Computes minimum horizon 12: 2 points
- Gives one drawback and one alternative: 2 points

---

## Exercise 6 — Why simulation success is insufficient

### Expected answer

Simulation success is conditional on the simulator's sensing, dynamics, controller, reset, and task assumptions. A policy may exploit unrealistic contacts, perfect state information, deterministic timing, narrow initial states, or simplified visual conditions.

Real deployment introduces dynamics mismatch, sensor noise, calibration drift, latency, actuator saturation, wear, unmodeled compliance, human presence, imperfect resets, and safety constraints. A high simulated return therefore demonstrates competence only under the benchmark distribution.

Evidence for deployment readiness should include held-out perturbation tests, system-identification checks, action and latency audits, safety-filter validation, slow hardware trials, failure categorization, intervention counts, and monitoring and fallback procedures.

### Suggested rubric

- Explains conditional nature of benchmark success: 3 points
- Identifies at least four sim-to-real gaps: 4 points
- Describes staged validation: 3 points

---

# Mathematical and Formulation Exercises

## Exercise 7 — Drawer-opening formulation

### Expected answer

One valid formulation is:

\[
o_t =
\left(
I_{t-L+1:t},
q_t,
\dot q_t,
g_t,
f_t
\right),
\]

where \(I\) is a visual history, \(q_t,\dot q_t\) are joint states, \(g_t\) is gripper state, and \(f_t\) is a wrist-force measurement.

A suitable action is

\[
a_t =
\left(
\Delta \mathbf{x}_t,
\Delta \boldsymbol{\omega}_t,
\Delta g_t
\right).
\]

A reward may be

\[
r_{t+1}
=
w_p\bigl(d_t-d_{t+1}\bigr)
+w_o \Delta \text{drawer}_{t+1}
+w_s \mathbb{I}_{\mathrm{success}}
-w_f c^{\mathrm{force}}_{t+1}
-w_a\|a_t\|_2^2.
\]

Termination:
- success when drawer displacement exceeds a threshold and grasp is released or stable;
- failure when force exceeds a hard limit, collision occurs, or the robot enters an unrecoverable configuration.

Truncation:
- time limit or external operator cutoff.

Constraints:
- joint limits;
- Cartesian workspace limits;
- force/torque threshold;
- collision avoidance;
- velocity or acceleration bounds.

Deployable observations:
- camera images, proprioception, gripper state, and measured force.

Training-only privileged variables:
- exact drawer pose, latch state, friction, simulator contact mode, and true object geometry.

### Suggested rubric

- Observation definition: 2 points
- Action definition: 2 points
- Reward: 2 points
- Success/termination/truncation: 2 points
- Constraints and privileged-variable distinction: 2 points

---

## Exercise 8 — Reward hacking when progress dominates success

### Expected answer

Suppose \(r^{\mathrm{progress}}_{t+1}=d_t-d_{t+1}\) rewards approaching a handle. If \(w_p\) is too large, the robot may repeatedly move toward and away from the handle to collect positive progress after resets or exploit noisy distance estimates. It may also hover near the target, collide aggressively to reduce measured distance, or optimize object motion without completing the grasp and opening sequence.

A corrected reward could use potential-based shaping:

\[
r_{t+1}
=
r^{\mathrm{task}}_{t+1}
+\gamma\Phi(s_{t+1})-\Phi(s_t)
-w_c c_{t+1},
\]

where \(\Phi\) is bounded and the task-success bonus dominates shaping. Another option is to cap cumulative progress reward and use stage completion bonuses with explicit preconditions.

The evaluation protocol should report sparse task success, violations, and shaped return separately. A policy should not be selected solely by shaped return.

### Teaching note

Students do not need to invoke the formal policy-invariance theorem, but they should understand why a difference of potentials is safer than an arbitrary dense reward.

---

## Exercise 9 — Offline RL with expert-only data

### Expected answer

Even if all demonstrations are successful, the dataset may cover only a narrow action manifold. A learned critic is trained on observed actions but the policy optimization step may query it at unseen actions. Function approximation can assign spuriously high values to these out-of-distribution actions. The actor then exploits critic error rather than learning a truly better behavior.

In

\[
\max_\theta
\mathbb{E}_{s\sim \mathcal D,\,
a\sim\pi_\theta(\cdot\mid s)}
[Q_\phi(s,a)]
-\lambda\Omega(\pi_\theta,\mathcal D),
\]

the regularizer \(\Omega\) discourages the policy from leaving the dataset support. Behavior constraints, conservative Q penalties, advantage-weighted regression, uncertainty estimates, and action-distance checks are possible controls.

Expert-only data can also make value ranking difficult because it contains little variation in outcome quality. In that case, BC may be as appropriate as offline RL unless additional failures or suboptimal trajectories are collected.

### Suggested rubric

- Identifies distributional shift: 3 points
- Explains critic extrapolation: 3 points
- Connects to regularizer: 2 points
- Notes expert-only ranking limitation or remedy: 2 points

---

## Exercise 10 — Effect of the execution horizon

### Expected answer

Increasing \(H_{\mathrm{exec}}\):
- preserves a longer coherent motion segment;
- reduces policy-query frequency and amortizes expensive inference;
- can improve smoothness;
- weakens feedback and delays correction;
- increases sensitivity to unexpected contacts and environment changes;
- can create stale commands.

Decreasing \(H_{\mathrm{exec}}\):
- increases feedback frequency;
- enables faster recovery and adaptation;
- requires more inference calls;
- may expose the policy to prediction jitter or reduce temporal commitment;
- may be infeasible when inference is slower than the low-level control loop.

The correct choice depends on task dynamics, inference latency, low-level control frequency, observation delay, and the reliability of the predicted tail of the chunk.

---

## Exercise 11 — Constrained mobile-robot objective

### Expected answer

Let \(R(\tau)\) reward goal completion and progress, and let \(C(\tau)\) count collisions or represent a collision indicator. Then

\[
J_R(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T-1}\gamma^t r_{t+1}
\right],
\]

\[
J_C(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T-1}\gamma_c^t c_{t+1}
\right],
\]

and the constrained problem is

\[
\max_\theta J_R(\theta)
\quad\text{subject to}\quad
J_C(\theta)\le d.
\]

For example, \(c_{t+1}=1\) on collision and \(0\) otherwise, while \(d=0.01\) could specify an expected discounted collision budget. If the desired quantity is episode-level collision probability, define

\[
J_C(\theta)
=
\Pr_{\tau\sim\pi_\theta}(\text{at least one collision})
\le d.
\]

The threshold must match the operational interpretation; expected discounted count and collision probability are not identical.

---

## Exercise 12 — Two-step action delay

### Expected answer

The delayed transition model is

\[
s_{t+1}\sim p(s_{t+1}\mid s_t,a_{t-2}).
\]

To recover a Markov state, augment it with the action queue:

\[
\tilde s_t = (s_t,a_{t-2},a_{t-1}).
\]

Then

\[
\tilde s_{t+1}
\sim
\tilde p(\tilde s_{t+1}\mid \tilde s_t,a_t).
\]

During simulation training, the environment should enqueue each selected action and apply the action from two control steps earlier. If real delay varies, randomize it over a plausible range. The observation should include recent actions or the controller command queue when available. Evaluation should include both nominal and larger delays.

The included `ReachingEnv` supports `action_delay_steps=2`.

---

# Implementation and Experiment-Design Exercises

## Exercise 13 — `RobotLearningEnv`

### Reference implementation

Use `src/ch13_guide/envs/reaching_env.py`.

The environment includes:
- Gymnasium-compatible `reset` and `step`;
- separate `terminated` and `truncated`;
- action scaling and saturation logging;
- a circular safety region and projection filter;
- reward-component logging;
- explicit success and timeout failure labels;
- optional observation noise, action noise, and action delay;
- deterministic seeding;
- RGB-array rendering.

### Required validation

Run:

```bash
pytest -q
```

The tests verify API compliance, seed reproducibility, success termination, time-limit truncation, and action-saturation logging.

### Instructor expectations

Students should explain:
- why success is `terminated`;
- why a time limit is `truncated`;
- why raw and executed actions are both logged;
- why the environment is pedagogical rather than a physics benchmark.

### Suggested rubric

- Correct Gymnasium API: 2 points
- Correct termination/truncation logic: 2 points
- Reward and success definition: 2 points
- Diagnostic logging: 2 points
- Tests and reproducibility: 2 points

---

## Exercise 14 — Behavior-cloning experiment for pick-and-place

### Reference design

#### Data collection

Collect teleoperated or scripted demonstrations containing:
- synchronized RGB or RGB-D observations;
- robot proprioception;
- end-effector and gripper commands;
- task and object identifiers;
- timestamps and controller frequency;
- success, failure, and intervention labels;
- corrective trajectories after imperfect approaches.

#### Split policy

Split by scene, object instance, operator, and initial condition. Do not randomly split adjacent frames because nearly identical temporal neighbors would leak across train and test sets.

A defensible split is:
- 70% training scenes and objects;
- 15% validation;
- 15% held-out test;
- one additional target-domain stress set.

#### Preprocessing

- resize and normalize images using training-set statistics;
- normalize proprioception and actions;
- preserve synchronization;
- represent rotations consistently;
- mask padded sequence positions;
- store normalization metadata.

#### Policy

Start with deterministic BC:

\[
\mathcal L_{\mathrm{BC}}
=
\frac{1}{N}\sum_i
\|\mu_\theta(o_i)-a_i\|_2^2.
\]

For multimodal action chunks, compare an action-chunking or diffusion model.

#### Evaluation

Report:
- closed-loop success;
- grasp success;
- placement success;
- intervention rate;
- safety-filter activations;
- action saturation;
- final position error;
- failure categories;
- inference latency.

Do not treat held-out mean-squared action error as sufficient.

### Executable notebook

See `notebooks/03_behavior_cloning_pick_place.ipynb`. It generates scripted reaching demonstrations, trains an MLP policy, and compares open-loop prediction error with closed-loop success.

---

## Exercise 15 — Sim-to-real perturbation suite for quadruped locomotion

### Reference suite

At minimum include:

1. **Mass and inertia shift**
   - Randomize body and leg masses.
   - Pass criterion: at least 90% of nominal success with no unsafe torque spikes.

2. **Ground friction**
   - Test low- and high-friction surfaces outside the training median.
   - Pass criterion: no fall and bounded foot-slip rate.

3. **Actuator strength and saturation**
   - Reduce available torque and alter motor gains.
   - Pass criterion: tracks commanded velocity within a specified error and respects actuator limits.

4. **Latency and jitter**
   - Add fixed and stochastic command delay.
   - Pass criterion: no instability up to the accepted deployment bound.

5. **Sensor noise and bias**
   - Perturb IMU, joint encoder, and velocity estimates.
   - Pass criterion: bounded tracking degradation and no repeated safety-trigger activation.

6. **Terrain**
   - Include slopes, height steps, and compliant patches.
   - Pass criterion: success across a predeclared range.

7. **External pushes**
   - Apply impulses at different gait phases.
   - Pass criterion: recovery without fall within a specified number of steps.

### Reporting

Use a perturbation matrix with:
- perturbation type;
- level;
- seed;
- success;
- tracking error;
- fall indicator;
- energy;
- peak torque;
- recovery time;
- safety intervention.

See `notebooks/04_sim_to_real_perturbation_suite.ipynb`.

---

## Exercise 16 — Offline-RL dataset for robotic pushing

### Reference design

The dataset should include:

#### Successful behavior
- direct pushes from many object poses;
- multiple valid contact points;
- varying friction and object shapes;
- short and long solutions;
- different approach directions.

#### Failed behavior
- missed contact;
- pushing the wrong side;
- object leaving the workspace;
- collisions;
- action saturation;
- insufficient force;
- overshoot.

#### Recovery behavior
- repositioning after missed contact;
- correcting lateral drift;
- re-approaching after occlusion;
- backing away from unsafe contact;
- recovering from object rotation.

#### Logged fields

At each step:
- observation;
- action;
- executed action;
- reward and cost components;
- next observation;
- terminated/truncated;
- task and object ID;
- timestamp and control frequency;
- camera calibration;
- robot embodiment;
- success/failure label;
- safety-filter activation;
- contact and force estimates;
- behavior-policy identifier and confidence, if available.

#### Dataset splits

Split by object instance, scene, and initial-condition region. Include a support audit and action histograms.

#### Offline-RL concern

A mixed-quality dataset is more useful for value learning than a narrow expert-only dataset because it provides outcome variation. However, low-quality data should be labeled and safety-critical failures should be reviewed before reuse.

See `notebooks/05_offline_rl_dataset_design.ipynb`.

---

## Exercise 17 — VLA manipulation benchmark

### Reference design

#### Task families
- object selection by color, shape, or category;
- spatial relations such as left of, inside, behind, or nearest;
- pick-and-place;
- tool or container use;
- refusal when preconditions are false.

#### Language templates
Examples:
- “Place the {color} {object} in the {container}.”
- “Move the object left of the {reference} onto the tray.”
- “Pick the {object} nearest the {reference}.”

#### Paraphrases
For each semantic task, create multiple surface forms:
- “Put the blue cup in the bowl.”
- “Move the blue cup into the bowl.”
- “The blue cup should end up inside the bowl.”

Paraphrases should be split so some templates appear only at test time.

#### Distractors
- same-category objects with different colors;
- visually similar objects;
- irrelevant objects;
- occluding objects;
- conflicting but infeasible targets.

#### Success conditions
Use physical state or robust perception to verify:
- correct object;
- correct relation or container;
- stable final placement;
- no safety violation;
- completion within time.

#### Failure taxonomy
- language misunderstanding;
- wrong-object grounding;
- spatial-relation error;
- grasp failure;
- control failure;
- action latency;
- unsafe motion;
- inability to refuse infeasible instruction;
- recovery failure.

#### Evaluation splits
- seen task/seen object;
- seen task/unseen object;
- unseen paraphrase;
- unseen viewpoint;
- distractor shift;
- target-robot shift.

Report closed-loop success, not only token or action prediction.

See `notebooks/06_vla_benchmark_design.ipynb`.

---

## Exercise 18 — Compare BC, online RL, and diffusion policy

### Reference task

Use tabletop pick-and-place with visual observations and Cartesian action chunks.

| Dimension | Behavior cloning | Online RL | Diffusion policy |
|---|---|---|---|
| Required data | Expert demonstrations | Environment interaction and reward | Demonstrated action sequences |
| Objective | Match expert actions | Maximize return | Model conditional action-chunk distribution |
| Main implementation | Dataset, supervised model, closed-loop evaluation | Simulator, actor-critic, replay/rollout, reward and safety | Windowed trajectories, noise schedule, denoiser, receding-horizon execution |
| Main strength | Stable, simple, no reward needed | Can improve recovery and optimize task success | Multimodal, temporally coherent actions |
| Main limitation | Covariate shift and no improvement beyond data | Unsafe and sample-intensive exploration | Iterative inference and no direct task optimization |
| Best use | Strong demonstrations and narrow deployment | Safe simulator or controlled fine-tuning | Multimodal manipulation demonstrations |

A professional comparison should control:
- observation and action interfaces;
- network capacity where possible;
- training data budget;
- wall-clock compute;
- evaluation seeds;
- success definition;
- safety-filter configuration.

A useful hybrid is BC or diffusion pretraining followed by prior-regularized or residual RL.

See `notebooks/07_method_comparison.ipynb`.

---

# Instructor grading summary

A suggested grading distribution is:

- Conceptual exercises: 6 × 5 points = 30
- Mathematical/formulation exercises: 6 × 8 points = 48
- Implementation/experiment exercises: 6 × 12 points = 72
- Total: 150 points

The instructor may scale the total to the course grading scheme.

# Recommended assignment sequence

1. Exercises 1–6 as reading and discussion questions.
2. Exercises 7–12 as formulation homework.
3. Exercise 13 as the common coding foundation.
4. Exercises 14–18 as group projects or rotating laboratories.
5. Require every implementation report to include failure analysis and reproducibility metadata.