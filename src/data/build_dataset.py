"""
Dataset builder: joins the four source tables into one district-day table.

Output: one row per (district, date) with raw measurements, terrain
attributes and the label. Feature engineering happens later, in
src/features — this module only assembles and cleans.

MISSING-VALUE POLICY (documented deliberately, per the proposal)
----------------------------------------------------------------
We do NOT blanket-fill everything with zero or the mean.
  - rainfall_mm  : short gaps (<= 2 days) are linearly interpolated
                   within a district; longer gaps stay NaN and the
                   affected rows are dropped. A missing rainfall day is
                   not a zero-rainfall day, and treating it as one
                   would fabricate dry spells.
  - river_level_m: interpolated the same way, limit 3 days — gauge
                   dropouts are common and levels are autocorrelated.
  - terrain      : static, must be complete. Missing values are an error.
A `was_imputed` flag is kept so the share of imputed rows can be
reported honestly.
"""

import pandas as pd

from src.config import CFG, ROOT
from src.data import loaders


def _interpolate_within_district(
    df: pd.DataFrame, column: str, limit: int
) -> pd.DataFrame:
    df = df.sort_values(["district", "date"]).copy()
    df[f"{column}_was_missing"] = df[column].isna()
    df[column] = (
        df.groupby("district", group_keys=False)[column]
        .apply(lambda s: s.interpolate(method="linear", limit=limit, limit_area="inside"))
    )
    return df


def build(save: bool = True) -> pd.DataFrame:
    tables = loaders.load_all()

    rainfall = _interpolate_within_district(tables["rainfall"], "rainfall_mm", limit=2)
    river = _interpolate_within_district(tables["river"], "river_level_m", limit=3)

    terrain = tables["terrain"]
    if terrain.isna().any().any():
        raise ValueError("Terrain table has missing values — it must be complete.")

    df = (
        rainfall
        .merge(river, on=["date", "district"], how="outer")
        .merge(tables["labels"], on=["date", "district"], how="left")
        .merge(terrain, on="district", how="left")
    )

    n_before = len(df)
    df = df.dropna(subset=["rainfall_mm", "river_level_m", "flood_occurred"])
    n_dropped = n_before - len(df)

    df["flood_occurred"] = df["flood_occurred"].astype(int)
    df = df.sort_values(["district", "date"]).reset_index(drop=True)

    imputed_share = (
        df["rainfall_mm_was_missing"] | df["river_level_m_was_missing"]
    ).mean()

    print(f"[dataset] {len(df):,} district-days "
          f"({df['district'].nunique()} districts, "
          f"{df['date'].min().date()} to {df['date'].max().date()})")
    print(f"[dataset] dropped {n_dropped:,} rows with unrecoverable gaps")
    print(f"[dataset] {imputed_share:.2%} of retained rows had an interpolated value")
    print(f"[dataset] positive class rate: {df['flood_occurred'].mean():.2%}")
    if loaders.is_fixture():
        print("[dataset] *** SOURCE IS SYNTHETIC FIXTURE — results are not reportable ***")

    if save:
        out = ROOT / CFG.data.processed
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"[dataset] wrote {out}")

    return df


if __name__ == "__main__":
    build()
