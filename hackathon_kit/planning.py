"""Approximate censored-demand learning and joint inventory planning.

This is a policy model, not privileged knowledge of the simulator demand law.
Poisson demand with a log-linear price effect is an explicit approximation.
"""
import math
from copy import deepcopy
import numpy as np


class DemandModel:
    def __init__(self, market, settings):
        self.settings = deepcopy(settings)
        self.anchor = math.sqrt(market["price_min"]) * math.sqrt(market["price_max"])
        levels, slopes = np.meshgrid(np.array([0.5, 1, 2, 4, 8]) * settings["demand_prior"],
                                     np.array([0.4, 0.7, 1, 1.6, 2.5]) * settings["price_elasticity"])
        self.levels, self.slopes = levels.ravel(), slopes.ravel()
        self.weights = np.full(len(self.levels), 1 / len(self.levels))
        self.updates = 0

    def rates(self, prices, competitor_price):
        prices = np.asarray(prices, dtype=float)
        if prices.ndim != 1 or not len(prices) or not np.all(np.isfinite(prices)) or np.any(prices <= 0):
            raise ValueError("Demand prices must be a nonempty vector of positive finite values")
        if competitor_price is not None and (not math.isfinite(competitor_price) or competitor_price < 0):
            raise ValueError("Invalid competitor price")
        reference = self.anchor if competitor_price is None else competitor_price
        delta = prices[None, :] - self.anchor - self.settings["competition_weight"] * (reference - self.anchor)
        return np.clip(self.levels[:, None] * np.exp(np.clip(-self.slopes[:, None] * delta, -30, 30)), 1e-9, 10000)

    def revenue_curve(self, prices, maximum_stock, competitor_price):
        """Optimal expected revenue and price for each available stock 0..maximum."""
        if type(maximum_stock) is not int or maximum_stock < 0:
            raise ValueError("Maximum stock must be a nonnegative integer")
        rates = self.rates(prices, competitor_price)
        log_pmf = -rates.copy()
        log_rates = np.log(rates)
        pmf = np.exp(log_pmf)
        cdf = pmf.copy()
        expected = np.zeros_like(rates)
        revenue = [0.0]
        selected = [float(prices[len(prices) // 2])]
        for stock in range(1, maximum_stock + 1):
            # Sum survival probabilities. Recovering PMFs from their logarithm
            # is essential: exp(-rate) underflows at high rates, while the PMF
            # around its mode remains substantial (e.g. rate=1000).
            expected = np.minimum(rates, expected + np.clip(1 - cdf, 0, 1))
            mean = self.weights @ expected
            variance = self.weights @ ((expected - mean) ** 2)
            optimistic = np.minimum(stock, mean + self.settings["demand_optimism"] * np.sqrt(variance))
            values = np.asarray(prices) * optimistic
            index = int(np.argmax(values))
            revenue.append(float(values[index]))
            selected.append(float(prices[index]))
            log_pmf += log_rates - math.log(stock)
            pmf = np.exp(log_pmf)
            cdf = np.minimum(1, cdf + pmf)
        return revenue, selected

    def update(self, price, competitor_price, sales, offered, censored):
        if type(sales) is not int or type(offered) is not int or not 0 <= sales <= offered:
            raise ValueError("Demand model needs integer sales <= offered stock")
        if type(censored) is not bool:
            raise ValueError("Demand censorship must be explicit")
        if offered == 0:
            return
        rates = self.rates([price], competitor_price)[:, 0]
        if censored:
            # Observe D >= sales, not D == sales. Even an equality stockout is censored.
            if sales == 0:
                return
            log_likelihood = poisson_log_survival(rates, sales)
        else:
            log_likelihood = sales * np.log(rates) - rates - math.lgamma(sales + 1)
        decay = self.settings["demand_decay"]
        prior = decay * self.weights + (1 - decay) / len(self.weights)
        log_weights = np.log(np.maximum(prior, 1e-300)) + log_likelihood
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        self.weights = weights / weights.sum()
        self.updates += 1

    def state_dict(self):
        return {"weights": self.weights.tolist(), "updates": self.updates}

    def load_state_dict(self, state):
        weights = np.asarray(state["weights"], dtype=float)
        if weights.shape != self.weights.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Invalid demand model weights")
        if not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12) or type(state["updates"]) is not int or state["updates"] < 0:
            raise ValueError("Invalid demand model state")
        self.weights = weights.copy()
        self.updates = state["updates"]


def poisson_log_survival(rates, threshold):
    """Stable log P(Poisson(rate) >= threshold), including very rare stockouts."""
    rates = np.asarray(rates, dtype=float)
    if type(threshold) is not int:
        raise ValueError("Poisson threshold must be an integer")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("Poisson rates must be finite and nonnegative")
    result = np.zeros_like(rates)
    if threshold <= 0:
        return result
    for index, rate in np.ndenumerate(rates):
        if rate == 0:
            result[index] = -math.inf
            continue
        if threshold > rate:
            # Sum tail relative to its first term: no subtraction of nearly equal numbers.
            term = total = 1.0
            k = threshold
            while True:
                k += 1
                term *= rate / k
                total += term
                if term <= total * 1e-15:
                    break
            result[index] = threshold * math.log(rate) - rate - math.lgamma(threshold + 1) + math.log(total)
        else:
            # Sum the lower tail backwards relative to its largest term. Starting
            # from exp(-rate) would underflow for high-rate observations.
            k = threshold - 1
            term = total = 1.0
            while k > 0:
                term *= k / rate
                total += term
                k -= 1
                if term <= total * 1e-15:
                    break
            log_cdf = (threshold - 1) * math.log(rate) - rate - math.lgamma(threshold) + math.log(total)
            cdf = math.exp(min(0, log_cdf))
            result[index] = math.log1p(-min(cdf, 1 - 1e-15))
    return result


def joint_plan(obs, market, settings, curves, input_prices, purchasing):
    """Bounded beam over integer production bundles; costs and capacity shared.

For purchasing, missing complementary inputs are explicitly costed. For actual
production all quantities must be feasible now. Beam limit controls complexity.
"""
    products = sorted(market["recipes"])
    resources = sorted(market["reserve"])
    if type(settings["planner_limit"]) is not int or settings["planner_limit"] < 1:
        raise ValueError("Planner beam width must be a positive integer")
    # Admissible suffix bounds account for shared capacity, ignoring resource
    # costs/budgets. Ranking by partial profit alone discards capacity needed by
    # more valuable later products, even in the simplest one-unit problem.
    maxima = {}
    for product in products:
        stock = obs.stock[product]
        maximum = max(0, min(obs.capacity, settings["target_stock"] - stock))
        maxima[product] = maximum
        values = curves[product][0]
        if len(values) <= stock + maximum or any(not math.isfinite(v) for v in values):
            raise ValueError("Revenue curve must cover all feasible stock quantities with finite values")
    capacity_limit = min(obs.capacity, sum(maxima.values()))
    suffix = [[0.0] * (capacity_limit + 1) for _ in range(len(products) + 1)]
    for i in range(len(products) - 1, -1, -1):
        product = products[i]
        stock = obs.stock[product]
        values = curves[product][0]
        for capacity in range(capacity_limit + 1):
            suffix[i][capacity] = max(
                values[stock + q] - values[stock] - q * market["production_cost"][product]
                + suffix[i + 1][capacity - q]
                for q in range(min(maxima[product], capacity) + 1))
    pruned = 0
    # (score, produced, raw requirements, processing cost, incremental revenue)
    states = [(0.0, {}, {r: 0 for r in resources}, 0.0, 0.0)]
    for i, product in enumerate(products):
        next_states = []
        stock = obs.stock[product]
        maximum = maxima[product]
        for _, quantities, used, processing, revenue in states:
            for q in range(min(maximum, obs.capacity - sum(quantities.values())) + 1):
                requirements = {r: used[r] + market["recipes"][product].get(r, 0) * q for r in resources}
                production_cost = processing + q * market["production_cost"][product]
                if not purchasing and any(requirements[r] > obs.raw[r] for r in resources):
                    continue
                missing_cost = sum(max(0, requirements[r] - obs.raw[r]) * input_prices[r] for r in resources)
                budget = obs.cash * settings["cash_fraction"] if purchasing else obs.cash
                if production_cost + missing_cost > budget + 1e-9:
                    continue
                if purchasing and obs.remaining_budget is not None and missing_cost > obs.remaining_budget + 1e-9:
                    continue
                gross = revenue + curves[product][0][stock + q] - curves[product][0][stock]
                basis = sum(min(requirements[r], obs.raw[r]) * obs.raw_cost[r] +
                            max(0, requirements[r] - obs.raw[r]) * input_prices[r] for r in resources)
                score = gross - production_cost - basis
                next_states.append((score, {**quantities, product: q}, requirements, production_cost, gross))
        # Deterministic tie-break: prefer fewer units/costs when expected profit ties.
        next_states.sort(key=lambda s: (s[0] + suffix[i + 1][capacity_limit - sum(s[1].values())],
                                        s[0], -sum(s[1].values()), -s[3]), reverse=True)
        pruned += max(0, len(next_states) - settings["planner_limit"])
        states = next_states[:settings["planner_limit"]]
    best = states[0]
    return {"score": best[0], "quantities": best[1], "requirements": best[2],
            "processing": best[3], "revenue": best[4], "exact": pruned == 0, "pruned_states": pruned}
