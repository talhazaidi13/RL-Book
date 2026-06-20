
import numpy as np
from gymnasium.utils.env_checker import check_env

from ch13_guide.envs.reaching_env import ReachingConfig, ReachingEnv


def test_environment_api():
    env = ReachingEnv()
    check_env(env, skip_render_check=True)


def test_reset_is_seed_reproducible():
    env = ReachingEnv()
    obs1, _ = env.reset(seed=7)
    obs2, _ = env.reset(seed=7)
    assert np.allclose(obs1, obs2)


def test_step_returns_five_values():
    env = ReachingEnv()
    env.reset(seed=0)
    result = env.step(np.zeros(2, dtype=np.float32))
    assert len(result) == 5


def test_success_terminates():
    config = ReachingConfig(success_radius=0.06)
    env = ReachingEnv(config)
    env.reset(
        seed=0,
        options={
            "initial_position": np.array([0.5, 0.5], dtype=np.float32),
            "goal": np.array([0.54, 0.5], dtype=np.float32),
        },
    )
    _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated
    assert not truncated
    assert info["is_success"]


def test_timeout_truncates():
    config = ReachingConfig(max_steps=1, success_radius=1e-6)
    env = ReachingEnv(config)
    env.reset(
        seed=0,
        options={
            "initial_position": np.array([-0.8, -0.8], dtype=np.float32),
            "goal": np.array([0.8, 0.8], dtype=np.float32),
        },
    )
    _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert not terminated
    assert truncated
    assert info["failure_type"] == "timeout"


def test_action_saturation_logged():
    env = ReachingEnv()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([4.0, -4.0], dtype=np.float32))
    assert info["action_saturated"]
