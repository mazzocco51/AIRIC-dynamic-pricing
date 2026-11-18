"""Regression tests for delayed decisions, geometry and checkpoint integrity."""
from copy import deepcopy
import json

import numpy as np
import pytest

from hackathon_kit.config import load_config
from hackathon_kit.learners import Learner


def make(kind="ucb", arms=None, **settings):
    cfg = load_config()["agent"]
    cfg.update(settings)
    return Learner(kind, [5, 10, 15] if arms is None else arms, cfg)


@pytest.mark.parametrize("kind", ["ucb", "swucb", "gpucb"])
def test_delayed_start_explores_all_arms_before_repeating(kind):
    learner = make(kind)
    rng = np.random.default_rng(21)
    first = [learner.choose(rng) for _ in range(3)]
    assert {d["arm"] for d in first} == {0, 1, 2}
    # Reverse delivery must update the arm originally chosen, never the latest arm.
    for decision in reversed(first):
        learner.update(decision, decision["arm"] * 100)
    assert learner.pending == {}
    assert learner._counts.tolist() == [1, 1, 1]


@pytest.mark.parametrize("kind", ["ucb", "swucb", "exp3ix", "conversion_ts", "gpucb"])
def test_checkpoint_preserves_pending_rng_and_next_choices(kind):
    learner = make(kind)
    rng = np.random.default_rng(71)
    pending = [learner.choose(rng) for _ in range(3)]
    learner.update(pending.pop(), 8, successes=2, trials=5)
    restored = make(kind)
    restored.load_state_dict(json.loads(json.dumps(learner.state_dict())))
    restored_rng = np.random.default_rng()
    restored_rng.bit_generator.state = deepcopy(rng.bit_generator.state)
    for original in reversed(pending):
        learner.update(original, -12, successes=1, trials=8)
        restored.update(original, -12, successes=1, trials=8)
    for _ in range(8):
        decision = learner.choose(rng)
        assert decision == restored.choose(restored_rng)
        learner.update(decision, 14, successes=4, trials=10)
        restored.update(decision, 14, successes=4, trials=10)
    assert learner.state_dict() == restored.state_dict()


def test_gp_posterior_uses_price_distances_not_arm_indices():
    learner = make("gpucb", arms=[5, 5.01, 100], gp_length=0.2, gp_noise=0.01, gp_beta=2.0)
    # One observed expensive arm: the two nearby cheap arms must have nearly
    # identical uncertainty, unlike an erroneous equally-spaced index kernel.
    state = {"clock": 1, "history": [[0, 2, 0.8, None, None]], "log_weights": [0, 0, 0]}
    learner.load_state_dict(state)
    scores = learner._gp_scores()
    grid = (np.array(learner.arms) - 5) / 95
    cross = np.exp(-0.5 * ((grid - 1) / 0.2) ** 2)
    expected = 0.5 + cross / 1.01 * 0.3 + 2 * np.sqrt(1 - cross ** 2 / 1.01)
    np.testing.assert_allclose(scores, expected)
    assert scores[0] == pytest.approx(scores[1], abs=1e-7)


def test_gp_tiny_noise_repeated_observations_remains_finite():
    learner = make("gpucb", arms=[5], gp_noise=1e-30)
    rng = np.random.default_rng(3)
    for _ in range(20):
        decision = learner.choose(rng)
        learner.update(decision, 10)
    assert np.isfinite(learner._gp_scores()).all()


def test_window_expires_delayed_feedback_by_choice_time():
    learner = make("swucb", window=3)
    rng = np.random.default_rng(2)
    decisions = [learner.choose(rng) for _ in range(6)]
    for decision in reversed(decisions):
        learner.update(decision, 10)
    assert {row[0] for row in learner.history} == {3, 4, 5}
    assert learner._counts.sum() == 3
    assert learner.pending == {}


@pytest.mark.parametrize("kind", ["ucb", "swucb", "conversion_ts", "gpucb"])
def test_cached_statistics_match_observed_history(kind):
    learner = make(kind, window=4, gp_history=4)
    rng = np.random.default_rng(8)
    for i in range(30):
        d = learner.choose(rng)
        learner.update(d, 0.1 * i, successes=2, trials=9)
        expected = np.zeros(3)
        for _, arm, reward, _, _ in learner.history:
            expected[arm] += reward
        np.testing.assert_allclose(learner._sums, expected, atol=1e-14)
        assert learner._counts.sum() == len(learner.history)


def test_state_does_not_alias_input_or_export_and_invalid_load_is_atomic():
    learner = make()
    learner.update(learner.choose(np.random.default_rng(3)), 5)
    exported = learner.state_dict()
    exported["history"][0][2] = 0.1
    assert learner.history[0][2] != 0.1
    source = learner.state_dict()
    clone = make()
    clone.load_state_dict(source)
    source["history"][0][2] = 0.1
    assert clone.history[0][2] != 0.1
    original = clone.state_dict()
    bad = deepcopy(original)
    bad["statistics"]["counts"][0] += 8
    with pytest.raises(ValueError, match="statistics"):
        clone.load_state_dict(bad)
    assert clone.state_dict() == original


def test_mismatched_delayed_arm_is_rejected_before_update():
    learner = make()
    decision = learner.choose(np.random.default_rng(4))
    state = learner.state_dict()
    wrong = {**decision, "arm": (decision["arm"] + 1) % 3}
    with pytest.raises(ValueError, match="arm differs"):
        learner.update(wrong, 10)
    assert learner.state_dict() == state


@pytest.mark.parametrize("kind", ["ucb", "swucb", "exp3ix", "gpucb", "conversion_ts"])
def test_duplicate_feedback_is_idempotent(kind):
    learner = make(kind)
    decision = learner.choose(np.random.default_rng(4))
    learner.update(decision, 10, successes=1, trials=2)
    state = learner.state_dict()
    learner.update(decision, 10, successes=1, trials=2)
    assert learner.state_dict() == state


@pytest.mark.parametrize("kind,arms,settings", [
    ("unknown", [1], {}), ("ucb", [], {}), ("ucb", [1, 1], {}),
    ("ucb", [float("nan")], {}), ("ucb", [1], {"reward_low": 1, "reward_high": 1}),
    ("swucb", [1], {"window": 0}), ("exp3ix", [1], {"gamma": 0}),
    ("gpucb", [1], {"gp_noise": float("nan")}),
])
def test_invalid_learner_settings_rejected_early(kind, arms, settings):
    with pytest.raises(ValueError):
        make(kind, arms, **settings)
