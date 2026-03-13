"""Price European call and put options with Black-Scholes."""

from quant_toolkit.analysis import black_scholes_price

params = dict(
    spot=100.0,
    strike=105.0,
    time_to_expiry=0.5,      # 6 months
    risk_free_rate=0.05,
    volatility=0.20,
)

call = black_scholes_price(**params, option_type="call")
put = black_scholes_price(**params, option_type="put")

print(f"Spot: ${params['spot']:.2f}  Strike: ${params['strike']:.2f}")
print(f"Time: {params['time_to_expiry']} yr  Rate: {params['risk_free_rate']:.1%}  Vol: {params['volatility']:.1%}")
print(f"Call price: ${call:.4f}")
print(f"Put price:  ${put:.4f}")
