import numpy as np

from ch2_rl_fundamentals import (
    ReplayBuffer,
    compute_q_targets,
    empirical_squared_loss,
    soft_target_update,
)


def test_exercise_11_targets_and_loss():
    q = np.array([1.2, 0.4, -0.1])
    r = np.array([1.0, 0.0, -1.0])
    d = np.array([1.0, 1.0, 0.0])
    q_next = np.array([2.0, 1.0, 0.5])
    targets = compute_q_targets(r, d, q_next, gamma=0.95)
    expected = np.array([2.9, 0.95, -1.0])
    assert np.allclose(targets, expected)
    assert np.isclose(empirical_squared_loss(q, targets), 1.3341666666666667)


def test_soft_update_limits():
    theta = np.array([2.0, 4.0])
    theta_bar = np.array([0.0, 2.0])
    assert np.allclose(soft_target_update(theta, theta_bar, 1.0), theta)
    assert np.allclose(soft_target_update(theta, theta_bar, 0.0), theta_bar)


def test_replay_buffer_capacity_and_sampling():
    buffer = ReplayBuffer(capacity=2, seed=0)
    buffer.add(0, 0, 0.0, 1, False)
    buffer.add(1, 1, 1.0, 2, False)
    buffer.add(2, 0, 2.0, 3, True)
    assert len(buffer) == 2
    batch = buffer.sample(1)
    assert len(batch) == 1
