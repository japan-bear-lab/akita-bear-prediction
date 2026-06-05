"""M2 のメイン処理。

1. 出没点を読み込み（SQLite が無ければ CKAN から取得）
2. (メッシュ × 週) パネル + リークフリー特徴量を構築
3. data/panel.parquet に保存
4. 診断（行数・正例率・特徴量統計）+ 時系列分割でベースライン評価（AUC / PR-AUC）

実行: python -m src.build_panel
"""

from __future__ import annotations

import sqlite3

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config
from .data_loader import load_kumadas
from .panel import FEATURES, build_panel

PANEL_PATH = config.DATA_DIR / "panel.parquet"
TEST_FRACTION = 0.2  # 直近 20% の週をテスト（時系列分割）


def load_points() -> pd.DataFrame:
    """SQLite に保存済みならそれを使い、無ければ CKAN から取得。"""
    if config.DB_PATH.exists():
        with sqlite3.connect(config.DB_PATH) as conn:
            df = pd.read_sql("SELECT * FROM kumadas", conn, parse_dates=["dt"])
        print(f"✓ SQLite から読込: {len(df):,} 点")
        return df
    print("SQLite 未生成 → CKAN から取得")
    return load_kumadas()


def time_split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    weeks = sorted(panel["week"].unique())
    cut = weeks[int(len(weeks) * (1 - TEST_FRACTION))]
    train = panel[panel["week"] < cut]
    test = panel[panel["week"] >= cut]
    return train, test, pd.Timestamp(cut)


def evaluate_baselines(test: pd.DataFrame) -> None:
    """特徴量を単独スコアとして使った時の弁別力（AUC / PR-AUC）。

    モデルを当てる前に「どの信号がどれだけ効くか」を見る。学習不要・リークなし。
    """
    y = test["y"].to_numpy()
    base_rate = y.mean()
    print(f"\n--- 時系列分割テスト（正例率 = {base_rate:.3%}）---")
    print(f"{'スコア':<22}{'ROC-AUC':>10}{'PR-AUC':>10}")
    for col in ["nbr8_lag1", "roll4_count", "lag1_count", "weeks_since_last"]:
        s = test[col].to_numpy()
        if col == "weeks_since_last":
            s = -s  # 直近ほど出没しやすい → 符号反転
        try:
            auc = roc_auc_score(y, s)
            ap = average_precision_score(y, s)
            print(f"{col:<22}{auc:>10.3f}{ap:>10.3f}")
        except ValueError:
            print(f"{col:<22}{'n/a':>10}{'n/a':>10}")
    print(f"(PR-AUC のベースライン = 正例率 {base_rate:.3%})")


def main() -> None:
    df = load_points()
    print("パネル構築中 ...")
    panel = build_panel(df)

    n_mesh = panel[["row", "col"]].drop_duplicates().shape[0]
    n_week = panel["week"].nunique()
    print("\n=== パネル診断 ===")
    print(f"行数            : {len(panel):,}  ( {n_mesh:,} メッシュ × {n_week} 週 )")
    print(f"期間            : {panel['week'].min():%Y-%m-%d} 〜 {panel['week'].max():%Y-%m-%d}")
    print(f"正例 (y=1)      : {panel['y'].sum():,}  ({panel['y'].mean():.3%})")
    print("\n--- 特徴量サマリ ---")
    print(panel[FEATURES].describe().round(2).to_string())

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    print(f"\n✓ パネル保存: {PANEL_PATH}")

    train, test, cut = time_split(panel)
    print(f"\n時系列分割: train < {cut:%Y-%m-%d} ({len(train):,}) / test >= ({len(test):,})")
    evaluate_baselines(test)
    print("\n✓ M2 完了。次は M3（LightGBM で翌週出没確率 + 時系列 CV）。")


if __name__ == "__main__":
    main()
