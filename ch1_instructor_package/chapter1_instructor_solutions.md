# Chapter 1 Instructor Solutions

This instructor guide provides solutions and teaching notes for the Chapter 1 exercises. Mathematical answers are explicit enough for grading, while design exercises include representative responses rather than a single required answer.

---

## Conceptual Exercises

### Exercise 1

**Solution.** Reinforcement learning is sequential because an action changes not only the immediate reward but also the future states, future actions, and future opportunities available to the agent. In supervised learning, each prediction is usually evaluated against a fixed target label, and the prediction normally does not change the future data distribution. In RL, the agent's behavior affects what data it will see later.

For example, an autonomous warehouse robot choosing a shortcut may reduce immediate travel time but increase collision risk or lead to congestion. A supervised model might predict whether an aisle is blocked, but the RL problem is deciding a sequence of movements that maximizes long-term performance.

A strong answer should mention delayed consequences, state transitions, and cumulative return.

---

### Exercise 2

| Quantity | Classification | Explanation |
|---|---|---|
| \(r_{t+1}\) | Immediate | Reward received after taking \(a_t\) in \(s_t\). |
| \(G_t\) | Long-term | Discounted sum of future rewards from time \(t\). |
| \(V^\pi(s)\) | Long-term | Expected return from state \(s\) under policy \(\pi\). |
| \(Q^\pi(s,a)\) | Long-term | Expected return after taking action \(a\) in state \(s\), then following \(\pi\). |
| \(\pi(a\mid s)\) | Immediate action-selection rule | Probability of selecting action \(a\) in state \(s\). |

Clarification: \(\pi(a\mid s)\) affects long-term outcomes, but the quantity itself describes immediate action selection.

---

### Exercise 3

**Solution.** The true physical state contains all information needed to predict future dynamics, such as exact pose, velocity, map, obstacle locations, battery level, and actuator status. The observation is the information available through sensors, such as camera images, lidar scans, wheel encoders, or a noisy pose estimate. The RL state representation is the input actually provided to the learning algorithm; it may be the observation itself or a processed feature vector.

The observation may fail to be Markov if it omits information needed to predict the next state distribution. A single noisy pose estimate may not reveal velocity, wheel slip, battery level, or whether an obstacle is moving. Then two identical observations can require different decisions because their hidden physical states differ.

---

### Exercise 4

| Method | Example | Justification |
|---|---|---|
| Supervised learning | Defect classification from inspection images | Labeled examples are available; each prediction does not directly affect future data. |
| Model predictive control | Drone stabilization near hover with a known dynamics model | A reliable model and constraints are available; short-horizon optimization is suitable. |
| Reinforcement learning | Robot navigation in a complex simulated warehouse | The task involves sequential decisions, delayed rewards, uncertainty, and interaction. |

A strong answer should avoid claiming that RL is automatically best.

---

## Mathematical Exercises

### Exercise 5

The episodic return is

\[
G_0 = r_1+\gamma r_2+\gamma^2 r_3.
\]

For \(\gamma=0\):

\[
G_0=-1+0(-1)+0^2(5)=-1.
\]

For \(\gamma=0.5\):

\[
G_0=-1+0.5(-1)+0.5^2(5)=-1-0.5+1.25=-0.25.
\]

For \(\gamma=0.9\):

\[
G_0=-1+0.9(-1)+0.9^2(5)=-1-0.9+4.05=2.15.
\]

As \(\gamma\) increases, later rewards receive more weight. The terminal reward matters little when \(\gamma=0\), but strongly influences the return when \(\gamma=0.9\).

---

### Exercise 6

\[
V^\pi(s)=\sum_a \pi(a\mid s)Q^\pi(s,a).
\]

Therefore,

\[
V^\pi(s)=0.7(4)+0.3(10)=2.8+3=5.8.
\]

---

### Exercise 7

The Bellman target is

\[
r+\gamma V^\pi(s')=2+0.8(5)=6.
\]

---

### Exercise 8

Start from

\[
V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s].
\]

Using \(G_t=r_{t+1}+\gamma G_{t+1}\),

\[
V^\pi(s)=\mathbb{E}_\pi[r_{t+1}+\gamma G_{t+1}\mid s_t=s].
\]

In state \(s\), the policy selects action \(a\) with probability \(\pi(a\mid s)\). After action \(a\), the environment produces \((s',r)\) with probability \(p(s',r\mid s,a)\). Thus,

\[
V^\pi(s)=
\sum_{a\in\mathcal{A}(s)}\pi(a\mid s)
\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma V^\pi(s')\right].
\]

The outer summation is the expectation over actions selected by the policy. The inner summation is the expectation over environment transitions and rewards.

---

### Exercise 9

For \(Q^*(s,a)\), the first action \(a\) is fixed:

\[
Q^*(s,a)=\mathbb{E}\left[G_t\mid s_t=s,a_t=a,\text{ optimal behavior afterward}\right].
\]

Using the return recursion,

\[
Q^*(s,a)=\sum_{s',r}p(s',r\mid s,a)\left[r+\gamma V^*(s')\right].
\]

At the successor state,

\[
V^*(s')=\max_{a'}Q^*(s',a').
\]

Therefore,

\[
Q^*(s,a)=\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma\max_{a'}Q^*(s',a')\right].
\]

The maximization is over \(a'\), not over the first action \(a\), because \(Q^*(s,a)\) already assumes that the first action has been specified. The maximization appears after the transition because the agent chooses the next action only after observing the next state.

---

### Exercise 10

\[
\delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t).
\]

\[
\delta_t=1+0.9(6)-3=3.4.
\]

\[
V(s_t)\leftarrow V(s_t)+\alpha\delta_t=3+0.1(3.4)=3.34.
\]

---

### Exercise 11

The SARSA target is

\[
r_{t+1}+\gamma Q(s_{t+1},a_{t+1})=-1+0.9(5)=3.5.
\]

The update is

\[
Q(s_t,a_t)\leftarrow 2+0.2(3.5-2)=2.3.
\]

---

### Exercise 12

The Q-learning target is

\[
r_{t+1}+\gamma\max_{a'}Q(s_{t+1},a')=-1+0.9(7)=5.3.
\]

The update is

\[
Q(s_t,a_t)\leftarrow 2+0.2(5.3-2)=2.66.
\]

Q-learning updates more strongly here because it uses the best estimated next action value \(7\), while SARSA uses the value \(5\) of the action actually selected. SARSA is on-policy; Q-learning is off-policy.

---

## Implementation and Analysis Exercises

### Exercise 13

**Reference implementation.** See `src/gridworld.py`.

**Key design requirements.**

- `terminated=True` means the task ended naturally, for example the robot reached the goal or entered a terminal hazard.
- `truncated=True` means the episode was stopped externally, for example because a maximum step limit was reached.
- These flags should be separate because a time-limit cutoff does not necessarily mean the underlying MDP reached a terminal state.

**Minimal usage.**

```python
from src.gridworld import GridWorld

env = GridWorld()
state = env.reset(seed=0)

for _ in range(20):
    action = env.action_space_sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    print(state, action, reward, next_state, terminated, truncated)
    state = next_state
    if terminated or truncated:
        break
```

---

### Exercise 14

**Reference implementation.** See `td0_policy_evaluation` in `src/algorithms.py`.

Values near the goal should become positive if the goal reward is large enough. Values near the hazard should become lower. Values propagate backward from terminal outcomes through repeated TD updates.

**Minimal usage.**

```python
from src.gridworld import GridWorld
from src.algorithms import td0_policy_evaluation, make_epsilon_greedy_policy

env = GridWorld()
policy = make_epsilon_greedy_policy(env, epsilon=0.2)
V = td0_policy_evaluation(env, policy, episodes=500, alpha=0.1, gamma=0.95)
env.print_values(V)
```

---

### Exercise 15

**Reference implementation.** See `sarsa_control` and `q_learning_control` in `src/algorithms.py`.

Expected qualitative comparison:

- SARSA updates toward the action actually selected by the exploratory behavior policy.
- Q-learning updates toward the greedy action value, regardless of which exploratory action was selected.
- In hazardous navigation tasks, SARSA may learn a more conservative route when exploration persists.
- Q-learning may learn a shorter greedy path if the optimal greedy route has high value, even while exploration occasionally causes risky behavior.

**Minimal usage.**

```python
from src.gridworld import GridWorld
from src.algorithms import sarsa_control, q_learning_control, greedy_path

env = GridWorld()
Q_sarsa = sarsa_control(env, episodes=2000, alpha=0.2, gamma=0.95, epsilon=0.1)
Q_q = q_learning_control(env, episodes=2000, alpha=0.2, gamma=0.95, epsilon=0.1)

print("SARSA path:", greedy_path(env, Q_sarsa))
print("Q-learning path:", greedy_path(env, Q_q))
```

---

## Autonomous-System Design Exercises

### Exercise 16

A possible state representation includes drone position, velocity, battery level, camera orientation, inspected target locations, obstacle proximity, wind estimate, and mission time. The action space may be continuous motor commands, velocity commands, waypoint choices, or camera-gimbal commands. The reward may include positive reward for inspecting required structures, penalties for collisions, penalties for excessive energy use, penalties for unsafe proximity, and a completion bonus. Termination may occur when all inspection targets are completed, the drone collides, battery becomes critically low, or a mission time limit is reached.

Poor reward design can cause undesirable behavior. If the reward only counts inspected targets, the drone may fly too close to structures or ignore battery constraints. If the reward penalizes time too strongly, it may choose unsafe shortcuts. If the reward gives dense reward for camera coverage without verifying inspection quality, the drone may collect low-quality images while appearing to complete the task.

---

### Exercise 17

Classical control or trajectory optimization may be preferable for low-level attitude stabilization, tracking a known reference trajectory, enforcing hard thrust or fuel constraints, and solving well-modeled orbital transfer problems. These components often have reliable dynamics models and strict safety requirements.

RL may be justified for high-level decision-making under uncertainty, adaptive maneuver selection, recovery from off-nominal conditions, or long-horizon policies where modeling all contingencies is difficult. Direct exploration on a real spacecraft would be unacceptable, so RL would need simulation, verification, constraints, and possibly combination with MPC or trajectory optimization.

---

### Exercise 18

Example: autonomous warehouse robot delivery.

- Episodic or continuing: episodic if each delivery order is treated as one episode; continuing if the robot operates indefinitely across many tasks.
- Finite or infinite horizon: finite horizon if there is a delivery deadline or maximum step count; infinite horizon if the objective is long-run operational efficiency.
- Fully or partially observed: usually partially observed because the robot may not know future human motion, hidden obstacles, or exact floor conditions.
- Model-based or model-free: model-based if the algorithm uses a map and transition model for planning; model-free if it learns values or policies only from sampled experience.
- Tabular or function approximation: tabular only for a very small grid; function approximation is required for lidar, camera images, continuous positions, or large maps.

A modeling ambiguity is whether the robot's map and localization system are accurate enough to make the state approximately Markov. Additional information needed before selecting an RL algorithm includes safety constraints, simulator availability, cost of exploration, action space, reward specification, and whether expert demonstrations are available.
