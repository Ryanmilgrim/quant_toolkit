import numpy as np
import pandas as pd

from quant_toolkit.returns import to_log_returns


def test_simple_series():
    s = pd.Series([0.01, 0.02, -0.005])
    result = to_log_returns(s)
    expected = np.log1p(s)
    pd.testing.assert_series_equal(result, expected)


def test_dataframe():
    df = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, -0.01]})
    result = to_log_returns(df)
    expected = np.log1p(df)
    pd.testing.assert_frame_equal(result, expected)


def test_masks_minus_one():
    s = pd.Series([0.01, -1.0, 0.02])
    result = to_log_returns(s)
    assert np.isnan(result.iloc[1])
    assert np.isfinite(result.iloc[0])
    assert np.isfinite(result.iloc[2])


def test_masks_below_minus_one():
    s = pd.Series([0.01, -1.5, 0.02])
    result = to_log_returns(s)
    assert np.isnan(result.iloc[1])
