"""
Tabular RL algorithms for Chapter 1 exercises.

Includes:
- TD(0) policy evaluation
- SARSA control
- Q-learning control
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple
import numpy as np

from gridworld import GridWorld, State

Policy = Callable[[State], int]


def initialize_values(env: GridWorld) -> Dict[State, float]:
    return {s: 0.0 for s in env.states()}


def initialize_q(env: GridWorld) -> Dict[State, np.ndarray]:
    return {s: np.zeros(env.n_actions, dtype=float) for s in env.states()}


def make_epsilon_greedy_policy(env: GridWorld, epsilon: float = 0.2) -> Policy:
    """Simple policy biased toward right/up but with exploration."""
    preferred = [1, 0]

    def policy(state: State) -> int:
        if np.random.rand() < epsilon:
            return env.action_space_sample()
        return int(np.random.choice(preferred))

    return policy


def make_epsilon_greedy_policy_from_q(env: GridWorld, Q: Dict[State, np.ndarray], epsilon: float) -> Policy:
    def policy(state: State) -> int:
        if np.random.rand() < epsilon:
            return env.action_space_sample()
        return int(np.argmax(Q[state]))

    return policy


def td0_policy_evaluation(
    env: GridWorld,
    policy: Policy,
    episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.95,
    seed: int = 0,
) -> Dict[State, float]:
    np.random.seed(seed)
    V = initialize_values(env)
    for ep in range(episodes):
        state = env.reset(seed=seed + ep)
        while True:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            bootstrap = 0.0 if terminated else V[next_state]
            td_error = reward + gamma * bootstrap - V[state]
            V[state] += alpha * td_error
            state = next_state
            if terminated or truncated:
                break
    return V


def sarsa_control(
    env: GridWorld,
    episodes: int = 2000,
    alpha: float = 0.2,
    gamma: float = 0.95,
    epsilon: float = 0.1,
    seed: int = 0,
) -> Dict[State, np.ndarray]:
    np.random.seed(seed)
    Q = initialize_q(env)
    for ep in range(episodes):
        state = env.reset(seed=seed + ep)
        policy = make_epsilon_greedy_policy_from_q(env, Q, epsilon)
        action = policy(state)
        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            if terminated:
                target = reward
            else:
                next_action = policy(next_state)
                target = reward + gamma * Q[next_state][next_action]
            Q[state][action] += alpha * (target - Q[state][action])
            if terminated or truncated:
                break
            state = next_state
            action = next_action
    return Q


def q_learning_control(
    env: GridWorld,
    episodes: int = 2000,
    alpha: float = 0.2,
    gamma: float = 0.95,
    epsilon: float = 0.1,
    seed: int = 0,
) -> Dict[State, np.ndarray]:
    np.random.seed(seed)
    Q = initialize_q(env)
    for ep in range(episodes):
        state = env.reset(seed=seed + ep)
        policy = make_epsilon_greedy_policy_from_q(env, Q, epsilon)
        while True:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            if terminated:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state])
            Q[state][action] += alpha * (target - Q[state][action])
            state = next_state
            if terminated or truncated:
                break
    return Q


def greedy_path(env: GridWorld, Q: Dict[State, np.ndarray], max_steps: int = 50) -> List[Tuple[State, str]]:
    state = env.reset(seed=123)
    path = []
    for _ in range(max_steps):
        if state not in Q:
            break
        action = int(np.argmax(Q[state]))
        action_name = env.action_names[action]
        path.append((state, action_name))
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        if terminated or truncated:
            path.append((state, "END"))
            break
    return path
