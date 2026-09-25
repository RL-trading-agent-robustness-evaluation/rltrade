"""Gate E2 Environment acceptance tests per Regulation V1 §19."""

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from src.data.adapter import build_canonical_dataset
from src.env.trading_env import RLTradingEnv


@pytest.fixture(scope="module")
def canonical_df():
    return build_canonical_dataset()


def test_gym_and_sb3_check_env(canonical_df):
    """Verifies that RLTradingEnv strictly passes Gymnasium and SB3 checkers."""
    env = RLTradingEnv(canonical_df)
    # Gymnasium check_env
    gym_check_env(env)
    # SB3 check_env
    sb3_check_env(env)


def test_observation_dimensions_and_finite(canonical_df):
    """Checks observation shape (17,) and ensures no NaNs or Infs occur."""
    env = RLTradingEnv(canonical_df)
    obs, info = env.reset(seed=42)

    assert obs.shape == (17,)
    assert np.isfinite(obs).all()

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
        assert obs.shape == (17,)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)


def test_seed_determinism(canonical_df):
    """Checks that identical seeds and action sequences produce exact same NAV trajectories."""
    env1 = RLTradingEnv(canonical_df)
    env2 = RLTradingEnv(canonical_df)

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    np.testing.assert_allclose(obs1, obs2)

    actions = [2, 2, 1, 0, 1, 2, 0, 1, 1, 2]
    for a in actions:
        o1, r1, t1, _, info1 = env1.step(a)
        o2, r2, t2, _, info2 = env2.step(a)
        assert r1 == pytest.approx(r2, abs=1e-10)
        assert info1["nav_close"] == pytest.approx(info2["nav_close"], abs=1e-6)
        assert info1["cash_post_trade"] == pytest.approx(info2["cash_post_trade"], abs=1e-6)
        np.testing.assert_allclose(o1, o2)
