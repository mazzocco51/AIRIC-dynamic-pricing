"""Serializable learners with explicit decision-time feedback and sufficient statistics."""
from copy import deepcopy
import math
import warnings
import numpy as np


class Learner:
    KINDS = {"ucb", "swucb", "exp3ix", "conversion_ts", "gpucb"}

    def __init__(self, kind, arms, settings, low=None, high=None):
        self.kind = kind
        self.arms = list(arms)
        self.cfg = deepcopy(settings)
        self.low = settings["reward_low"] if low is None else low
        self.high = settings["reward_high"] if high is None else high
        if kind not in self.KINDS:
            raise ValueError(f"Unknown learner: {kind}")
        if not self.arms or any(isinstance(a, bool) or not math.isfinite(a) for a in self.arms):
            raise ValueError("Arms must be nonempty finite numbers")
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("Arms must be distinct")
        if not math.isfinite(max(self.arms) - min(self.arms)):
            raise ValueError("Arm range is too wide")
        if not math.isfinite(self.low) or not math.isfinite(self.high) or not self.low < self.high:
            raise ValueError("Invalid reward bounds")
        if not math.isfinite(self.high - self.low):
            raise ValueError("Reward interval is too wide")
        if kind == "swucb" and (type(settings["window"]) is not int or settings["window"] < 1):
            raise ValueError("Sliding window must be a positive integer")
        if kind == "exp3ix" and any(not math.isfinite(settings[k]) or settings[k] <= 0 for k in ("eta", "gamma")):
            raise ValueError("EXP3-IX eta and gamma must be positive and finite")
        if kind == "gpucb":
            if any(not math.isfinite(settings[k]) or settings[k] <= 0 for k in ("gp_length", "gp_noise", "gp_beta")):
                raise ValueError("GP length, noise and beta must be positive and finite")
            if type(settings["gp_history"]) is not int or settings["gp_history"] < 1:
                raise ValueError("GP history must be a positive integer")
        self.history = []  # [decision index, arm, normalized reward, successes, trials]
        self.log_weights = np.zeros(len(self.arms))
        self.clock = 0
        self.pending = {}
        self._counts = np.zeros(len(self.arms))
        self._sums = np.zeros(len(self.arms))
        self._successes = np.zeros(len(self.arms))
        self._trials = np.zeros(len(self.arms))

    @property
    def requirements(self):
        return {"feedback": "successes_trials" if self.kind == "conversion_ts" else "realized_reward",
                "reward_range": [self.low, self.high]}

    def normalize(self, reward):
        if not math.isfinite(reward):
            raise ValueError("Non-finite reward")
        if reward < self.low or reward > self.high:
            warnings.warn(f"Reward {reward} outside declared [{self.low}, {self.high}]; clipped", RuntimeWarning)
        reward = min(self.high, max(self.low, reward))
        return float((reward - self.low) / (self.high - self.low))

    def _accumulate(self, row, sign=1):
        _, arm, reward, successes, trials = row
        self._counts[arm] += sign
        self._sums[arm] += sign * reward
        if self.kind == "conversion_ts":
            self._successes[arm] += sign * successes
            self._trials[arm] += sign * trials

    def _expire(self, cutoff):
        retained = []
        for row in self.history:
            if row[0] < cutoff:
                self._accumulate(row, -1)
            else:
                retained.append(row)
        self.history = retained
        self.pending = {t: arm for t, arm in self.pending.items() if t >= cutoff}
        self._sums[self._counts == 0] = 0.0

    @staticmethod
    def _argmax_random(values, rng):
        return int(rng.choice(np.flatnonzero(values == np.max(values))))

    def _gp_scores(self):
        """Posterior on actual arm locations, including irregular rounded grids."""
        sample = self.history[-self.cfg["gp_history"]:]
        prices = np.asarray(self.arms, dtype=float)
        span = float(np.ptp(prices))
        grid = (prices - prices.min()) / span if span else np.zeros(len(prices))
        x = grid[np.array([row[1] for row in sample], dtype=int)]
        y = np.array([row[2] for row in sample])
        length = self.cfg["gp_length"]
        kernel = np.exp(-0.5 * ((x[:, None] - x) / length) ** 2)
        cross = np.exp(-0.5 * ((x[:, None] - grid) / length) ** 2)
        noise = max(self.cfg["gp_noise"], np.finfo(float).eps * max(1, len(x)))
        chol = np.linalg.cholesky(kernel + noise * np.eye(len(x)))
        # Multiple RHS: mean and variance share one factorization/solve.
        solved = np.linalg.solve(chol, np.column_stack((y - 0.5, cross)))
        mean = 0.5 + solved[:, 1:].T @ solved[:, 0]
        variance = np.maximum(0, 1 - (solved[:, 1:] ** 2).sum(axis=0))
        return mean + self.cfg["gp_beta"] * np.sqrt(variance)

    def choose(self, rng, unit_cost=0.0):
        if not math.isfinite(unit_cost):
            raise ValueError("Unit cost must be finite")
        t, k = self.clock, len(self.arms)
        if self.kind == "swucb":
            self._expire(t - self.cfg["window"])
        probability = None
        if self.kind in ("ucb", "swucb"):
            outstanding = np.bincount(list(self.pending.values()), minlength=k)
            effective = self._counts + outstanding
            unseen = np.flatnonzero(effective == 0)
            if len(unseen):
                arm = int(rng.choice(unseen))
            elif not self._counts.any():
                arm = int(rng.choice(np.flatnonzero(effective == effective.min())))
            else:
                # Pending pulls reserve exploration slots without inventing rewards.
                mean = np.divide(self._sums, self._counts,
                                 out=np.full(k, 0.5), where=self._counts > 0)
                horizon = self.cfg["window"] if self.kind == "swucb" else t + 1
                scores = mean + np.sqrt(2 * np.log(max(2, horizon)) / effective)
                arm = self._argmax_random(scores, rng)
        elif self.kind == "exp3ix":
            weights = np.exp(self.log_weights - np.max(self.log_weights))
            probabilities = weights / weights.sum()
            arm = int(rng.choice(k, p=probabilities))
            probability = float(probabilities[arm])
        elif self.kind == "conversion_ts":
            conversion = rng.beta(1 + self._successes, 1 + self._trials - self._successes)
            arm = self._argmax_random((np.array(self.arms) - unit_cost) * conversion, rng)
        else:
            if not self.history:
                remaining = np.array([i for i in range(k) if i not in self.pending.values()])
                if not len(remaining):
                    remaining = np.arange(k)
                midpoint = min(self.arms) + (max(self.arms) - min(self.arms)) / 2
                distances = np.abs(np.array(self.arms)[remaining] - midpoint)
                arm = int(rng.choice(remaining[distances == distances.min()]))
            else:
                arm = self._argmax_random(self._gp_scores(), rng)
        self.clock += 1
        self.pending[t] = arm
        return {"arm": arm, "time": t, "probability": probability}

    def update(self, decision, reward, successes=None, trials=None, censored=False):
        arm, t = decision["arm"], decision["time"]
        if type(arm) is not int or not 0 <= arm < len(self.arms):
            raise ValueError("Unknown arm")
        if type(t) is not int or not 0 <= t < self.clock:
            raise ValueError("Unknown decision time")
        if t not in self.pending:
            return  # Already consumed (including censored/expired feedback).
        if t in self.pending and self.pending[t] != arm:
            raise ValueError("Feedback arm differs from the issued decision")
        normalized = self.normalize(reward)
        if self.kind == "conversion_ts":
            if censored:
                self.pending.pop(t, None)
                return
            if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials:
                raise ValueError("Conversion TS requires integer successes <= trials")
        if self.kind == "exp3ix":
            p = decision["probability"]
            if p is None or not math.isfinite(p) or not 0 < p <= 1:
                raise ValueError("Original action probability required for delayed feedback")
            loss = 1 - normalized
            decrement = self.cfg["eta"] * loss / (p + self.cfg["gamma"])
            if not math.isfinite(decrement) or not math.isfinite(self.log_weights[arm] - decrement):
                raise ValueError("EXP3-IX update overflow; reduce eta or increase gamma")
            self.log_weights[arm] -= decrement
            self.log_weights -= np.max(self.log_weights)
        self.pending.pop(t, None)
        if self.kind == "swucb":
            self._expire(self.clock - self.cfg["window"])
            if t < self.clock - self.cfg["window"]:
                return
        row = [t, arm, normalized, successes, trials]
        self.history.append(row)
        self._accumulate(row)
        if self.kind == "gpucb":
            self.history.sort(key=lambda row: row[0])
            while len(self.history) > self.cfg["gp_history"]:
                self._accumulate(self.history.pop(0), -1)

    def state_dict(self):
        return deepcopy({"history": self.history, "log_weights": self.log_weights.tolist(), "clock": self.clock,
                         "pending": {str(t): arm for t, arm in self.pending.items()},
                         "statistics": {"counts": self._counts.tolist(), "sums": self._sums.tolist(),
                                        "successes": self._successes.tolist(), "trials": self._trials.tolist()}})

    def load_state_dict(self, state):
        state = deepcopy(state)
        clock = state["clock"]
        if type(clock) is not int or clock < 0:
            raise ValueError("Invalid learner clock")
        weights = np.array(state["log_weights"], dtype=float)
        if weights.shape != (len(self.arms),) or not np.isfinite(weights).all():
            raise ValueError("Invalid learner log weights")
        history = state["history"]
        for row in history:
            if not isinstance(row, list) or len(row) != 5:
                raise ValueError("Invalid learner history row")
            t, arm, reward, s, n = row
            if type(t) is not int or not 0 <= t < clock or type(arm) is not int or not 0 <= arm < len(self.arms):
                raise ValueError("Invalid learner history decision")
            if not math.isfinite(reward) or not 0 <= reward <= 1:
                raise ValueError("Invalid normalized history reward")
            if self.kind == "conversion_ts" and (type(s) is not int or type(n) is not int or not 0 <= s <= n):
                raise ValueError("Invalid historical conversion counts")
        pending = {}
        for key, arm in state.get("pending", {}).items():
            if not ((type(key) is int) or (type(key) is str and key.isdecimal() and str(int(key)) == key)):
                raise ValueError("Invalid pending decision identifier")
            time = int(key)
            if time in pending:
                raise ValueError("Duplicate pending learner decisions")
            pending[time] = arm
        times = [row[0] for row in history]
        if len(set(times)) != len(times) or set(times) & set(pending):
            raise ValueError("Duplicate/overlapping learner decisions in checkpoint")
        if any(not 0 <= t < clock or type(arm) is not int or not 0 <= arm < len(self.arms) for t, arm in pending.items()):
            raise ValueError("Invalid pending learner decision")
        # Build statistics locally so failed validation cannot partially load state.
        counts = np.zeros(len(self.arms))
        sums = np.zeros(len(self.arms))
        successes = np.zeros(len(self.arms))
        trials = np.zeros(len(self.arms))
        for _, arm, reward, s, n in history:
            counts[arm] += 1
            sums[arm] += reward
            if self.kind == "conversion_ts":
                successes[arm] += s
                trials[arm] += n
        arrays = [counts, sums, successes, trials]
        cached = state.get("statistics")
        if cached is not None:
            stored = [np.asarray(cached[name], dtype=float) for name in ("counts", "sums", "successes", "trials")]
            if any(x.shape != (len(self.arms),) or not np.isfinite(x).all() or not np.allclose(x, y, rtol=1e-10, atol=1e-10)
                   for x, y in zip(stored, arrays)):
                raise ValueError("Learner statistics disagree with history")
            arrays = [x.copy() for x in stored]
        self.history, self.log_weights, self.clock, self.pending = history, weights, clock, pending
        self._counts, self._sums, self._successes, self._trials = arrays

