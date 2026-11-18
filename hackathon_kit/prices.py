"""Decimal tick arithmetic at the serialization boundary (no fixed precision)."""
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext
import math


def _decimal(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Prices and ticks must be finite numbers")
    return Decimal(str(value))


def tick_interval(low, high, tick):
    lo, hi, step = _decimal(low), _decimal(high), _decimal(tick)
    if step <= 0 or lo > hi:
        raise ValueError("Invalid tick interval")
    with localcontext() as ctx:
        # Float exponents span roughly +/-308. Preserve integral quotients even
        # when values and ticks are at opposite ends of that range.
        ctx.prec = 700
        first = int((lo / step).to_integral_value(rounding=ROUND_CEILING))
        last = int((hi / step).to_integral_value(rounding=ROUND_FLOOR))
    if first > last:
        raise ValueError("No legal price tick inside observed bounds")
    return first, last


def round_to_tick(price, low, high, tick):
    first, last = tick_interval(low, high, tick)
    value, step = _decimal(price), _decimal(tick)
    with localcontext() as ctx:
        ctx.prec = 700
        index = int((value / step).to_integral_value(rounding=ROUND_HALF_EVEN))
        rounded = float(min(last, max(first, index)) * step)
    if not low <= rounded <= high:
        raise ValueError("Rounded price cannot be represented inside the bounds")
    return rounded


def is_tick_aligned(price, tick):
    value, step = _decimal(price), _decimal(tick)
    if step <= 0:
        return False
    with localcontext() as ctx:
        ctx.prec = 700
        quotient = value / step
        nearest = quotient.to_integral_value(rounding=ROUND_HALF_EVEN)
        candidate = float(nearest * step)
    # Permit only floating representation error, not a material fraction of a tick.
    return abs(price - candidate) <= min(abs(tick) * 1e-7, max(math.ulp(price), math.ulp(candidate)))
