"""
Run Chapter 1 gridworld experiments.

Usage:
    python src/run_experiments.py
"""

from gridworld import GridWorld
from algorithms import (
    make_epsilon_greedy_policy,
    td0_policy_evaluation,
    sarsa_control,
    q_learning_control,
    greedy_path,
)


def main() -> None:
    env = GridWorld(slip_prob=0.05)

    print("=== TD(0) Policy Evaluation ===")
    policy = make_epsilon_greedy_policy(env, epsilon=0.2)
    V = td0_policy_evaluation(env, policy, episodes=1000, alpha=0.1, gamma=0.95)
    env.print_values(V)

    print("\n=== SARSA Control ===")
    Q_sarsa = sarsa_control(env, episodes=3000, alpha=0.2, gamma=0.95, epsilon=0.1)
    print(env.render_policy(Q_sarsa))
    print("Greedy path:", greedy_path(env, Q_sarsa))

    print("\n=== Q-Learning Control ===")
    Q_q = q_learning_control(env, episodes=3000, alpha=0.2, gamma=0.95, epsilon=0.1)
    print(env.render_policy(Q_q))
    print("Greedy path:", greedy_path(env, Q_q))


if __name__ == "__main__":
    main()
