
import pandas as pd


def add_features(df):
    """
    Create additional features for customer churn prediction.

    Parameters:
        df: Input customer DataFrame

    Returns:
        DataFrame with additional engineered features
    """

    df = df.copy()

    # Group customers based on their tenure
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months"
        ]
    )

    return df
