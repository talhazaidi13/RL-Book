# Instructor Guide: Chapter 2 Exercise Solutions

This guide provides solution sketches, grading expectations, and small implementation notes for Chapter 2. The level of detail is intended for instructors and teaching assistants rather than students seeing the exercises for the first time.

---

## Conceptual Exercises

### Exercise 1
**Prompt.** Explain the difference between model-free and model-based reinforcement learning without using Bellman-equation derivations. Give one example where each family is attractive.

**Solution.** Model-free reinforcement learning learns a policy, value function, or actor--critic pair directly from interaction data, without explicitly learning a predictive model of the environment. Its central question is: *which action should the agent take, or how good is an action, given the data it has observed?* It is attractive when the environment is difficult to model but interaction in simulation is cheap, such as Atari-style discrete control, simulated locomotion, or benchmark continuous-control tasks.

Model-based reinforcement learning learns or uses a model of environment dynamics, rewards, or latent transitions. The learned model can be used to predict consequences, perform planning, generate imagined rollouts, or improve a policy with fewer real interactions. It is attractive when real interaction is expensive and a useful model can be learned or specified, such as robotic control with a simulator, spacecraft guidance with accurate dynamics, or industrial control with a digital twin.

**Expected answer elements.**
- Model-free: no explicit dynamics model; learns policy/value directly from experience.
- Model-based: uses known or learned predictive model.
- Model-free example: DQN in discrete games; PPO/SAC in simulation.
- Model-based example: MPC with learned dynamics; PETS/MBPO/world-model methods.

---

### Exercise 2
**Prompt.** A value-based method, a policy-gradient method, and an actor--critic method are applied to the same discrete-action task. What does each method learn, and how does each select actions?

**Solution.** A value-based method learns a value function, usually an action-value function $Q_\theta(s,a)$. It selects actions by maximizing the learned values, for example $a \in \arg\max_a Q_\theta(s,a)$, sometimes with exploration such as $\epsilon$-greedy action selection.

A policy-gradient method learns a parameterized policy $\pi_\theta(a\mid s)$ directly. It selects actions by sampling from the learned categorical distribution or by choosing the highest-probability action at evaluation time.

An actor--critic method learns both a policy, called the actor, and a value function, called the critic. In a discrete-action task, the actor may output a categorical action distribution while the critic estimates $V_\phi(s)$, $Q_\phi(s,a)$, or an advantage estimate. The actor selects actions; the critic supplies learning signals that reduce variance or improve update quality.

**Common mistake.** Students often say actor--critic selects actions using the critic. That can be true in some greedy value-improvement variants, but in standard actor--critic the actor is the action-selection mechanism and the critic evaluates or shapes the update.

---

### Exercise 3
**Prompt.** Why is offline RL not simply off-policy RL with a large replay buffer? Explain using dataset support and extrapolation error.

**Solution.** Off-policy online RL with replay usually continues collecting new experience while learning. If the learner discovers that it lacks data in an important region of state--action space, it may eventually visit that region and add new transitions to the buffer. Offline RL has a fixed dataset. It cannot repair missing coverage by interacting with the environment.

Dataset support is the set of state--action pairs represented with sufficient density in the data. When a learned policy chooses actions outside this support, the value function or model must extrapolate. Extrapolation error occurs when the learner assigns unreliable values to unsupported actions because those actions were rarely or never observed. Offline RL therefore needs conservative, behavior-constrained, uncertainty-aware, or policy-regularized objectives that discourage unsupported actions.

**Expected answer elements.**
- Replay buffer in online off-policy RL is continually refreshed.
- Offline dataset is fixed.
- Dataset support limits what can be reliably evaluated or improved.
- Extrapolation error can make unsupported actions appear falsely attractive.

---

### Exercise 4
**Prompt.** Compare deterministic and stochastic policies for continuous control. When might a stochastic policy be preferable even if the final deployed controller should be nearly deterministic?

**Solution.** A deterministic policy maps each state to one action, $a=\mu_\theta(s)$. It is efficient at execution time and often appropriate for smooth low-level control after training. A stochastic policy represents a distribution, such as a Gaussian $\pi_\theta(a\mid s)$, and samples actions during training or under uncertainty.

A stochastic policy can be preferable during learning because it supports exploration, entropy regularization, robustness to uncertainty, and policy-gradient estimation. It can also represent multiple plausible actions in ambiguous states. After training, the deployed controller may use the mean action or a low-variance version of the policy, making behavior nearly deterministic while still benefiting from stochastic exploration during learning.

**Examples.** SAC uses a stochastic policy during training and often evaluates using the mean action. PPO often uses stochastic policies for exploration and stable policy improvement.

---

### Exercise 5
**Prompt.** Explain why partial observability is not solved automatically by using a larger neural network. What architectural or data-design changes might be needed?

**Solution.** Partial observability means the current observation $o_t$ does not contain all information needed to infer the underlying state. A larger feedforward network cannot recover information that is absent from its input. It may memorize correlations, but it cannot infer hidden variables such as velocity, intent, damage, wind disturbance, or unobserved object pose unless the input contains enough temporal or auxiliary evidence.

Useful changes include stacking recent observations, adding recurrent networks, using attention over history, maintaining a learned belief state, incorporating state estimators or filters, adding informative sensors, or changing the data collection process to include histories. The key is not merely increasing capacity, but providing memory or information that helps infer hidden state.

**Expected answer elements.**
- Capacity is not equivalent to observability.
- Need history, recurrence, filtering, attention, belief-state estimation, or better sensors.

---

## Mathematical and Implementation Exercises

### Exercise 6
**Prompt.** Let $Q_\theta(s,a)$ be trained with the target $y=r+\gamma\max_{\tilde{a}}Q_{\bar{\theta}}(s^{+},\tilde{a})$. Write the empirical squared loss over a minibatch of size $m$ and identify which quantities are treated as targets during the gradient update.

**Solution.** For minibatch $\mathcal{B}=\{(s_i,a_i,r_i,s_i^+)\}_{i=1}^m$, the empirical squared loss is

$$
\mathcal{L}(\theta)=\frac{1}{m}\sum_{i=1}^{m}\left(Q_\theta(s_i,a_i)-y_i\right)^2,
\qquad
 y_i=r_i+\gamma\max_{\tilde{a}}Q_{\bar{\theta}}(s_i^+,\tilde{a}).
$$

During the gradient update, $Q_\theta(s_i,a_i)$ is differentiated with respect to $\theta$. The rewards, next states, actions, sampled transitions, and target-network values $Q_{\bar\theta}$ are treated as fixed target quantities. In most implementations $y_i$ is detached from the computation graph, so gradients do not flow into $\bar\theta$ through the target.

**Code reference.** See `compute_q_targets` and `empirical_squared_loss` in `src/ch2_rl_fundamentals/ch2_utils.py`.

---

### Exercise 7
**Prompt.** Suppose a soft target update uses $\bar{\theta}\leftarrow \tau\theta+(1-\tau)\bar{\theta}$. What happens qualitatively as $\tau\rightarrow 1$? What happens as $\tau\rightarrow 0$?

**Solution.** As $\tau\rightarrow 1$, the target parameters rapidly become equal to the online parameters. In the limiting case $\tau=1$, the update is $\bar\theta\leftarrow\theta$, which is a hard copy. The target network is no longer slow-moving and provides little stabilization.

As $\tau\rightarrow 0$, the target network changes very slowly. In the limiting case $\tau=0$, $\bar\theta$ is unchanged. This can stabilize targets, but if it is too slow the target becomes stale and may slow learning.

**Instructor note.** The best value is algorithm- and problem-dependent. The point is that target networks trade responsiveness against stability.

---

### Exercise 8
**Prompt.** A rollout buffer stores $\log\pi_\theta(a_t\mid s_t)$ at data-collection time. Why should an implementation store these log probabilities instead of recomputing them only after the policy has been updated?

**Solution.** In on-policy methods, the probability of an action under the policy that collected the data is part of the training objective. If the policy is updated and the log probability is recomputed afterward, the implementation uses the probability under a different policy. This corrupts importance ratios, clipped objectives, entropy diagnostics, and policy-gradient estimates.

For PPO-style methods, for example, the ratio

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
$$

requires the denominator from the data-collection policy. Storing the old log probability allows a stable computation of $\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)$ even after the current network changes.

---

### Exercise 9
**Prompt.** Consider a learned dynamics model $\hat{s}^{+}_\phi=f_\phi(s,a)$. Write a supervised loss for training the model from transition data. What additional term would be needed if the reward model is also learned?

**Solution.** For transitions $\mathcal{B}=\{(s_i,a_i,r_i,s_i^+)\}_{i=1}^m$, a basic continuous-state supervised dynamics loss is

$$
\mathcal{L}_{\mathrm{dyn}}(\phi)=\frac{1}{m}\sum_{i=1}^{m}\left\| f_\phi(s_i,a_i)-s_i^+\right\|_2^2.
$$

If the reward is also learned using $\hat r_\phi(s,a)$, add a reward-prediction term:

$$
\mathcal{L}_{\mathrm{model}}(\phi)=\frac{1}{m}\sum_{i=1}^{m}
\left[
\left\| f_\phi(s_i,a_i)-s_i^+\right\|_2^2
+
\lambda_r\left(\hat r_\phi(s_i,a_i)-r_i\right)^2
\right].
$$

The coefficient $\lambda_r$ balances state-prediction and reward-prediction scales.

**Extension.** For stochastic dynamics, a negative log-likelihood can replace squared error.

---

### Exercise 10
**Prompt.** Give two reasons why a value estimate can increase during training while the true policy performance decreases.

**Solution.** First, overestimation bias can make the value function assign high values to actions that are not actually good. This is common when bootstrapping and maximization interact with approximation error. The critic may become more optimistic even as the policy exploits critic errors.

Second, the value estimate may be accurate only under the training distribution but unreliable under the distribution induced by the updated policy. If policy updates move the agent into poorly covered states or actions, the value function can extrapolate incorrectly. Other valid reasons include reward hacking, evaluation mismatch, critic loss decreasing under a replay distribution that does not match deployment, or reduced exploration causing premature convergence to a poor policy.

**Expected answer elements.** Two of the following are sufficient: overestimation, distribution shift, extrapolation error, nonstationary targets, reward misspecification, evaluation mismatch, insufficient exploration.

---

### Exercise 11
**Prompt.** A minibatch contains three transitions. The current network outputs for the selected actions are $Q_\theta(s_i,a_i)=(1.2,0.4,-0.1)$. The rewards are $(1,0,-1)$, the nonterminal indicators are $(1,1,0)$, and the maximum target-network values at the next states are $(2.0,1.0,0.5)$. With $\gamma=0.95$ and

$$
y_i=r_i+\gamma d_i\max_{\tilde{a}}Q_{\bar{\theta}}(s_i^+,\tilde{a}),
$$

compute the three targets and the empirical squared loss $\frac{1}{3}\sum_i(Q_\theta(s_i,a_i)-y_i)^2$.

**Solution.** The targets are

$$
y_1 = 1 + 0.95(1)(2.0)=2.90,
$$

$$
y_2 = 0 + 0.95(1)(1.0)=0.95,
$$

$$
y_3 = -1 + 0.95(0)(0.5)=-1.00.
$$

The prediction errors are

$$
1.2-2.9=-1.7,\qquad 0.4-0.95=-0.55,\qquad -0.1-(-1.0)=0.9.
$$

The squared errors are

$$
2.89,\qquad 0.3025,\qquad 0.81.
$$

Therefore,

$$
\mathcal{L}=\frac{2.89+0.3025+0.81}{3}=1.334166\ldots \approx 1.3342.
$$

**Minimal Python check.**

```python
import numpy as np
q = np.array([1.2, 0.4, -0.1])
r = np.array([1.0, 0.0, -1.0])
d = np.array([1.0, 1.0, 0.0])
q_next = np.array([2.0, 1.0, 0.5])
gamma = 0.95

y = r + gamma * d * q_next
loss = np.mean((q - y) ** 2)
print(y)     # [ 2.9   0.95 -1.  ]
print(loss)  # 1.3341666666666667
```

---

## Algorithm Selection Exercises

### Exercise 12
**Prompt.** A mobile robot has a good simulator, continuous velocity commands, sparse rewards, and occasional collisions in early exploration. Which algorithm families would you consider first, and what safety precautions would you add?

**Solution.** Because the action space is continuous, actor--critic continuous-control methods are natural candidates. PPO, SAC, or TD3 could be considered depending on the simulator budget and desired exploration behavior. Since rewards are sparse, the instructor should expect students to mention reward shaping, curriculum learning, goal-conditioned learning, demonstrations, intrinsic motivation, or hierarchical methods. Since a good simulator is available, model-based RL or planning-assisted methods may also be attractive, especially for sample efficiency.

Safety precautions should include training in simulation before deployment, collision penalties or constraints, safety shields, action limits, velocity and acceleration bounds, domain randomization, early termination on unsafe states, conservative real-world fine-tuning, and evaluation under rare but plausible obstacle configurations.

**Good answer.** SAC or PPO with a safety layer, curriculum, simulation pretraining, and careful real-world validation.

**Weak answer.** “Use DQN” without addressing continuous actions or safety.

---

### Exercise 13
**Prompt.** A warehouse routing problem has a discrete action space and millions of logged trajectories from a rule-based controller, but no safe online exploration. Should the first approach be online DQN, behavior cloning, offline RL, or model-based planning? Justify your choice.

**Solution.** Online DQN should not be the first choice because safe online exploration is not available. Behavior cloning is a strong first baseline because there are many logged trajectories from a rule-based controller and it is simple, stable, and easy to validate. Offline RL is also a reasonable next step if reward labels are available and the goal is to improve beyond the rule-based controller. If a reliable warehouse dynamics model or simulator exists, model-based planning can also be considered, but the prompt emphasizes logged data rather than a verified model.

A practical sequence is: start with behavior cloning as a supervised baseline, evaluate it offline and in simulation if available, then try conservative offline RL or behavior-regularized offline RL to improve decisions while staying near dataset support.

**Expected answer.** Behavior cloning first, offline RL second if rewards and sufficient support exist. Avoid online DQN as the first approach.

---

### Exercise 14
**Prompt.** A UAV must make real-time control decisions under wind disturbances and communication delays. Which task properties from Table~\ref{tab:ch2-task-to-algorithm} matter most for algorithm selection?

**Solution.** The most important task properties are continuous action space, real-time execution constraints, partial observability, safety constraints, dynamics uncertainty, communication delay, and simulator availability. Wind disturbances create stochasticity and model mismatch; communication delays create delayed observations or delayed control. If low-level control commands are continuous, actor--critic continuous-control methods, robust control, or model-based control should be considered. If deployment is safety-critical, constrained RL, shields, and robust evaluation are needed.

Partial observability matters because the current observation may not reveal wind state or delayed actuator effects. Recurrent policies, history windows, state estimators, or filters may be needed. Real-time constraints matter because a method requiring expensive online planning at every step may be impractical unless planning is bounded and predictable.

---

### Exercise 15
**Prompt.** A spacecraft guidance problem has accurate simulated dynamics but no possibility of real-world trial-and-error. Explain why benchmark return alone is insufficient for evaluating a learned policy.

**Solution.** Benchmark return is insufficient because spacecraft guidance is safety-critical and physically constrained. A policy with high return may violate thrust limits, miss terminal constraints, use excessive fuel, enter unsafe regions, produce unstable trajectories, or fail under small perturbations. Return also hides distributional information: two policies with the same average return may differ greatly in worst-case terminal error, constraint satisfaction, fuel consumption, robustness, and sensitivity to initialization.

Evaluation should include terminal position and velocity error, fuel or $\Delta v$, time of flight, constraint violations, robustness across initial conditions, sensitivity to model perturbations, failure rate, uncertainty margins, and comparison to classical guidance or trajectory-optimization baselines. Since real-world exploration is impossible, validation in high-fidelity simulation and stress testing are essential.

---

### Exercise 16
**Prompt.** A multi-agent traffic-signal control system improves average throughput but increases worst-case delay for a minority of intersections. Which taxonomy branches and evaluation practices are relevant for diagnosing this result?

**Solution.** Multi-agent RL is directly relevant because each intersection may be controlled by a separate agent or by a coordinated policy. Safe, constrained, robust, or risk-sensitive RL is also relevant because average throughput has improved at the expense of worst-case service. If the system uses learned coordination or decentralized execution, centralized training with decentralized execution may be relevant.

Evaluation should report not only mean throughput but also worst-case delay, delay percentiles, queue lengths, fairness across intersections, emergency-vehicle delays, spatial distribution of failures, robustness to demand changes, and performance under rare congestion patterns. The result may indicate that the reward function overweights global throughput while underweighting local fairness or tail risk.

**Expected answer elements.** Multi-agent RL, safe/risk-sensitive RL, fairness/tail metrics, distributional reporting rather than average-only evaluation.

---

### Exercise 17
**Prompt.** Choose one algorithm from each major branch in Figure~\ref{fig:ch2-taxonomy-tree}. For each, state the data it uses, the object it learns, one strength, and one failure mode.

**Solution.** There are many acceptable answers. One representative solution is below.

| Major branch | Example algorithm | Data used | Object learned | Strength | Failure mode |
|---|---|---|---|---|---|
| Model-free value-based RL | DQN | Online/off-policy transitions in replay | Action-value function $Q_\theta(s,a)$ | Effective for discrete actions | Instability/overestimation; poor fit for continuous actions |
| Model-free policy-gradient RL | PPO | On-policy rollout batches | Stochastic policy and value baseline | Robust and widely used | Sample inefficient; sensitive to reward scaling and rollout settings |
| Continuous-control actor--critic | SAC | Off-policy replay data | Stochastic actor and soft Q critics | Strong exploration and continuous-control performance | Hyperparameter sensitivity; critic extrapolation errors |
| Model-based RL | MBPO | Real transitions plus short model rollouts | Dynamics model and policy/value functions | Better sample efficiency | Model bias and compounding rollout error |
| Latent world models | Dreamer-style method | Sequences of observations/actions/rewards | Latent dynamics, reward/value/policy | Learns from imagined latent rollouts | Latent model can miss task-relevant details |
| Imitation learning | Behavior cloning | Expert state-action demonstrations | Supervised policy | Simple and stable baseline | Covariate shift under compounding errors |
| Offline RL | CQL | Fixed reward-labeled dataset | Conservative Q function and policy | Reduces unsupported-action overestimation | Can be over-conservative or limited by dataset quality |
| Hierarchical RL | Options or Option-Critic | Interaction data with temporal abstraction | Skills/options and high-level selection | Helps long-horizon tasks | Poor skill discovery; nonstationary lower-level skills |
| Multi-agent RL | MAPPO or MADDPG | Multi-agent trajectories | Coordinated policies and/or centralized critics | Handles coordination | Nonstationarity, credit assignment, fairness failures |
| Safe/constrained RL | CPO or Lagrangian constrained RL | Transitions with reward and cost signals | Policy satisfying expected constraints | Makes constraints explicit | Over-conservatism or constraint violations under shift |
| Preference-based RL/agents | PPO-style RLHF or DPO | Preference comparisons or ranked responses | Reward model or preference-aligned policy | Aligns behavior with human preferences | Reward hacking, preference noise, distribution shift |

**Grading note.** Do not require the same algorithm choices. Grade whether the student correctly identifies data, learned object, strength, and failure mode for each selected branch.
