import pandas as pd
import numpy as np
import os
from include.logic.data_preparation import (
    inject_gaps_dynamically,
    apply_discrete_adapter,
)


def test_dynamic_gap_injection():
    """Ensures the gap injection math produces exactly the requested missingness."""

    # Create a dummy dataframe of 100 rows
    df_mock = pd.DataFrame(
        {
            "time": range(100),
            "cpu_usage": np.random.rand(100),
            "lat50_ms": np.random.randint(100, 500, 100),
        }
    )

    target_cols = ["cpu_usage", "lat50_ms"]

    # Request 20% missingness (20 rows) in blocks of 5
    df_injected = inject_gaps_dynamically(
        df_mock, target_cols, missing_ratio=0.2, block_size=5
    )

    assert "is_gap" in df_injected.columns
    assert df_injected["is_gap"].sum() == 20, "Failed to inject exactly 20 gaps."
    assert df_injected["cpu_usage"].isna().sum() == 20, (
        "Failed to NaN the target columns."
    )


def test_discrete_adapter(tmp_path):
    """Ensures the adapter fixes duplicate seconds and fills missing seconds with NaNs."""

    # Create mock data with a duplicate and a missing second
    df_mock = pd.DataFrame(
        {
            "time": [10, 10, 11, 13],
            "cpu_usage": [0.5, 0.8, 0.9, 1.2],
        }
    )

    # Save to a temporary pytest directory
    mock_path = os.path.join(tmp_path, "mock.parquet")
    df_mock.to_parquet(mock_path)

    # Run the adapter
    out_path = apply_discrete_adapter(mock_path, str(tmp_path))
    df_discrete = pd.read_parquet(out_path)

    # Assertions
    assert len(df_discrete) == 4, "Should be length 4: times 10, 11, 12, 13"
    assert df_discrete["time"].tolist() == [10, 11, 12, 13], "Time grid is broken"

    # Check duplicate handling (Should keep the LAST duplicate: 0.8)
    assert df_discrete.loc[df_discrete["time"] == 10, "cpu_usage"].iloc[0] == 0.8

    # Check gap filling (Second 12 should be NaN)
    assert pd.isna(df_discrete.loc[df_discrete["time"] == 12, "cpu_usage"].iloc[0])
