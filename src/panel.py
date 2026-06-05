"""M2: 出没点 → (メッシュ × 週) パネルデータ化。

設計（HANDOFF 2 章）:
- 目的変数 y: 各 1km メッシュ m について、翌週 t に出没が 1 件以上あるか（二値）
- 空間単位: 標準地域メッシュ 3 次（≒1km 四方）。jismesh で付与
- 時間単位: 週次（月曜起点で統一）
- 負例不均衡対策: 対象を「出没のあったメッシュ + 8 近傍バッファ」に限定
- データリーク厳禁: 特徴量は必ず「その週より前」の情報のみ

整数グリッド (row, col) = (floor(lat*120), floor(lon*80)) を持つ。
3 次メッシュは緯度 1/120 度・経度 1/80 度刻みなので、これは 3 次メッシュと
1 対 1 対応し、8 近傍が row±1 / col±1 で取れる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import jismesh.utils as ju
except ImportError as e:  # pragma: no cover
    raise ImportError("jismesh が必要です: pip install jismesh") from e

# 3 次メッシュの刻み（度）
LAT_UNIT = 1.0 / 120.0  # 30 秒
LON_UNIT = 1.0 / 80.0   # 45 秒

# 学習に使う開始年（2021 以前はオープンデータ収録カバレッジの人工物のため除外）
START_YEAR = 2022

# 8 近傍オフセット
NEIGHBOR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def assign_mesh(df: pd.DataFrame) -> pd.DataFrame:
    """出没点に 3 次メッシュ ID と整数グリッド (row, col) を付与する。"""
    df = df.copy()
    df["mesh_id"] = [
        ju.to_meshcode(lat, lon, 3) for lat, lon in zip(df["lat"], df["lon"])
    ]
    df["row"] = np.floor(df["lat"] / LAT_UNIT).astype(int)
    df["col"] = np.floor(df["lon"] / LON_UNIT).astype(int)
    return df


def to_week_start(dt: pd.Series) -> pd.Series:
    """日時 → その週の月曜 0:00（週キー）。"""
    d = dt.dt.normalize()
    return d - pd.to_timedelta(d.dt.dayofweek, unit="D")


def build_panel(df: pd.DataFrame, start_year: int = START_YEAR) -> pd.DataFrame:
    """出没点 DataFrame → (メッシュ × 週) パネル + リークフリー特徴量。"""
    df = assign_mesh(df)
    df = df[df["dt"].dt.year >= start_year].copy()
    df["week"] = to_week_start(df["dt"])

    # --- 週次・メッシュ別の出没件数 ---
    counts = (
        df.groupby(["row", "col", "week"])
        .size()
        .reset_index(name="count_w")
    )
    # mesh_id 対応表（row,col → mesh_id）
    mesh_map = df.drop_duplicates("mesh_id")[["row", "col", "mesh_id"]]

    # --- 対象メッシュ集合: 出没メッシュ + 8 近傍バッファ ---
    occurred = counts[["row", "col"]].drop_duplicates()
    cells = {(int(r), int(c)) for r, c in occurred.itertuples(index=False)}
    buffered = set(cells)
    for r, c in cells:
        for dr, dc in NEIGHBOR8:
            buffered.add((r + dr, c + dc))
    target = pd.DataFrame(sorted(buffered), columns=["row", "col"])

    # --- 週グリッド（最初の週 〜 最後の週、月曜起点で連続）---
    weeks = pd.date_range(df["week"].min(), df["week"].max(), freq="W-MON")
    week_df = pd.DataFrame({"week": weeks})

    # --- 全格子（対象メッシュ × 全週）を直積 ---
    panel = target.merge(week_df, how="cross")
    panel = panel.merge(counts, on=["row", "col", "week"], how="left")
    panel["count_w"] = panel["count_w"].fillna(0).astype(int)
    panel["y"] = (panel["count_w"] >= 1).astype(int)

    # week を整数 index 化（シフト・近傍結合用）
    week_index = {w: i for i, w in enumerate(weeks)}
    panel["w"] = panel["week"].map(week_index)
    panel = panel.sort_values(["row", "col", "w"]).reset_index(drop=True)

    # --- ラグ特徴（すべて t より前の情報のみ）---
    g = panel.groupby(["row", "col"], sort=False)["count_w"]
    shifted = g.shift(1)  # 前週件数（グループ内、index 整列）
    panel["lag1_count"] = shifted.fillna(0)
    # 過去 4 週合計（t-4 .. t-1）。シフト済み系列をグループ内で rolling する
    panel["roll4_count"] = (
        panel.assign(_s=shifted)
        .groupby(["row", "col"], sort=False)["_s"]
        .rolling(4, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    ).fillna(0)
    # 最終出没からの経過週数（出没なしは大きめの値で頭打ち）
    panel["weeks_since_last"] = _weeks_since_last(panel)

    # --- 近傍 8 メッシュの前週合計（リークフリー）---
    panel["nbr8_lag1"] = _neighbor_prev_week_sum(panel)

    # --- 前年同週(±2週)の件数: T-54 .. T-50（リークフリー。前年は必ず過去）---
    panel["yoy_lag"] = (
        panel.assign(_s=g.shift(50))
        .groupby(["row", "col"], sort=False)["_s"]
        .rolling(5, min_periods=1).sum()
        .reset_index(level=[0, 1], drop=True)
    ).fillna(0)

    # --- メッシュの過去出没率（expanding・その週より前だけで集計）---
    gy = panel.groupby(["row", "col"], sort=False)["y"]
    cum_pos = gy.cumsum().shift(1)        # 当該週より前の「出没した週数」
    cum_n = gy.cumcount()                 # 当該週より前の週数 (0,1,2,...)
    panel["mesh_rate"] = (cum_pos / cum_n.replace(0, np.nan)).fillna(0)

    # --- 季節性 ---
    panel["month"] = panel["week"].dt.month
    panel["woy"] = panel["week"].dt.isocalendar().week.astype(int)

    panel = panel.merge(mesh_map, on=["row", "col"], how="left")
    return panel


def _weeks_since_last(panel: pd.DataFrame, cap: int = 52) -> pd.Series:
    """各 (mesh, week) で、直近の出没から何週経過したか（その週は含めない）。"""
    out = np.full(len(panel), cap, dtype=int)
    last_seen: dict[tuple[int, int], int] = {}
    rows = panel[["row", "col", "w", "count_w"]].to_numpy()
    for i in range(len(rows)):
        r, c, w, cnt = int(rows[i, 0]), int(rows[i, 1]), int(rows[i, 2]), int(rows[i, 3])
        key = (r, c)
        if key in last_seen:
            out[i] = min(cap, w - last_seen[key])
        if cnt >= 1:
            last_seen[key] = w
    return pd.Series(out, index=panel.index)


def _neighbor_prev_week_sum(panel: pd.DataFrame) -> pd.Series:
    """近傍 8 メッシュの「前週」出没件数合計（データリークなし）。"""
    base = panel[["row", "col", "w", "count_w"]]
    acc = pd.Series(0.0, index=panel.index)
    key = panel[["row", "col", "w"]].copy()
    for dr, dc in NEIGHBOR8:
        # 近傍セル (row+dr, col+dc) の前週 (w-1) の count を自セルに寄せる
        src = base.rename(columns={"count_w": "c"}).copy()
        src["row"] = src["row"] - dr
        src["col"] = src["col"] - dc
        src["w"] = src["w"] + 1  # その週から見た「前週」に合わせる
        merged = key.merge(src, on=["row", "col", "w"], how="left")
        acc = acc + merged["c"].fillna(0).to_numpy()
    return acc


# mesh_rate は時系列分割で逆効果だったため特徴量から除外（列は診断用に残す）。
# yoy_lag（前年同週±2週）はほぼ中立だが「前年同期」を明示的に考慮するため採用。
FEATURES = ["lag1_count", "roll4_count", "nbr8_lag1", "weeks_since_last",
            "yoy_lag", "month", "woy"]
