import math

import pytest

from quant_toolkit.analysis import black_scholes_price


def test_call_price_basic():
    price = black_scholes_price(
        spot=100, strike=100, time_to_expiry=1.0,
        risk_free_rate=0.05, volatility=0.2,
    )
    assert price > 0
    assert isinstance(price, float)


def test_put_price_basic():
    price = black_scholes_price(
        spot=100, strike=100, time_to_expiry=1.0,
        risk_free_rate=0.05, volatility=0.2, option_type="put",
    )
    assert price > 0
    assert isinstance(price, float)


def test_put_call_parity():
    params = dict(spot=100, strike=105, time_to_expiry=0.5,
                  risk_free_rate=0.03, volatility=0.25)
    call = black_scholes_price(**params, option_type="call")
    put = black_scholes_price(**params, option_type="put")
    # C - P = S - K * exp(-rT)
    lhs = call - put
    rhs = params["spot"] - params["strike"] * math.exp(-params["risk_free_rate"] * params["time_to_expiry"])
    assert abs(lhs - rhs) < 1e-10


def test_deep_in_the_money_call():
    price = black_scholes_price(
        spot=200, strike=50, time_to_expiry=1.0,
        risk_free_rate=0.05, volatility=0.2,
    )
    intrinsic = 200 - 50 * math.exp(-0.05)
    assert price >= intrinsic - 1e-6


def test_invalid_inputs():
    with pytest.raises(ValueError):
        black_scholes_price(spot=-1, strike=100, time_to_expiry=1, risk_free_rate=0.05, volatility=0.2)
    with pytest.raises(ValueError):
        black_scholes_price(spot=100, strike=100, time_to_expiry=0, risk_free_rate=0.05, volatility=0.2)
    with pytest.raises(ValueError):
        black_scholes_price(spot=100, strike=100, time_to_expiry=1, risk_free_rate=0.05, volatility=-0.1)


def test_invalid_option_type():
    with pytest.raises(ValueError, match="option_type"):
        black_scholes_price(spot=100, strike=100, time_to_expiry=1, risk_free_rate=0.05, volatility=0.2, option_type="straddle")
