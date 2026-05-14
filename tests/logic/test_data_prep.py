import pandas as pd
import numpy as np
from include.logic.data_preparation import inject_gaps_dynamically


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
