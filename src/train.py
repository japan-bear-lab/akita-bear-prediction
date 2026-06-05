"""M3: 翌週出没確率モデル + 「次に出没しやすいエリア」ランキング。

- モデル: sklearn HistGradientBoostingClassifier（LightGBM 同等の GBDT。libomp 不要）
- 評価: 時系列分割（過去で学習 → 未来で検証）。AUC / PR-AUC を M2 ベースラインと比較
- 予測: 最新週の「次の週」について各メッシュの出没確率を算出しランキング
- 時間帯: 過去の目撃時刻分布からエリア別の出没しやすい時間帯を付与

実行: python -m src.train
（先に `python -m src.build_panel` で data/panel.parquet を作っておくこと）
"""

from __future__ import annotations

import json
import re
import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config
from .data_loader import load_kumadas
from .panel import FEATURES, LAT_UNIT, LON_UNIT, NEIGHBOR8, assign_mesh

PANEL_PATH = config.DATA_DIR / "panel.parquet"
PRED_JS_PATH = config.WEB_DIR / "predictions.js"
TEST_FRACTION = 0.2
TOP_MESH = 20
TOP_CITY = 15

# 時間帯ビン
TIME_BANDS = [
    (4, 7, "早朝(4-7)"), (7, 10, "朝(7-10)"), (10, 16, "日中(10-16)"),
    (16, 19, "夕方(16-19)"), (19, 23, "夜(19-23)"),
]


def time_band(hour: int) -> str:
    for lo, hi, label in TIME_BANDS:
        if lo <= hour < hi:
            return label
    return "深夜(23-4)"


def load_points() -> pd.DataFrame:
    if config.DB_PATH.exists():
        with sqlite3.connect(config.DB_PATH) as conn:
            return pd.read_sql("SELECT * FROM kumadas", conn, parse_dates=["dt"])
    return load_kumadas()


# ---------- 学習・評価 ----------

def time_split(panel: pd.DataFrame):
    weeks = sorted(panel["week"].unique())
    cut = weeks[int(len(weeks) * (1 - TEST_FRACTION))]
    return panel[panel["week"] < cut], panel[panel["week"] >= cut], pd.Timestamp(cut)


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, random_state=0,
    )


def pos_weight(y: pd.Series) -> np.ndarray:
    """不均衡対策の sample_weight（正例を neg/pos 倍に）。"""
    pos = max(int(y.sum()), 1)
    neg = len(y) - pos
    w = neg / pos
    return np.where(y.to_numpy() == 1, w, 1.0)


def train_and_eval(panel: pd.DataFrame) -> HistGradientBoostingClassifier:
    train, test, cut = time_split(panel)
    clf = make_model()
    clf.fit(train[FEATURES], train["y"], sample_weight=pos_weight(train["y"]))
    p = clf.predict_proba(test[FEATURES])[:, 1]
    auc = roc_auc_score(test["y"], p)
    ap = average_precision_score(test["y"], p)
    print(f"\n=== M3 モデル評価（時系列分割: test >= {cut:%Y-%m-%d}）===")
    print(f"  正例率(test)     : {test['y'].mean():.3%}")
    print(f"  ROC-AUC          : {auc:.3f}   (M2 ベスト単独 0.780)")
    print(f"  PR-AUC           : {ap:.3f}   (M2 ベスト単独 0.168 / 無情報 {test['y'].mean():.3f})")
    print(f"  → PR-AUC リフト  : 正例率の約 {ap / test['y'].mean():.1f} 倍")
    return clf


# ---------- 「次の週」予測 ----------

def next_week_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """最新週の次の週 T について、各候補メッシュの特徴量を組み立てる。"""
    last_w = int(panel["w"].max())
    T = last_w + 1
    target_date = pd.Timestamp(panel["week"].max()) + pd.Timedelta(days=7)

    cells = panel[["row", "col"]].drop_duplicates().reset_index(drop=True)
    cw = panel[["row", "col", "w", "count_w"]]

    # lag1: 最新週の件数
    lag1 = (cw[cw.w == last_w].set_index(["row", "col"])["count_w"])
    # roll4: 最新4週合計
    roll4 = (cw[cw.w.between(last_w - 3, last_w)].groupby(["row", "col"])["count_w"].sum())
    # nbr8: 近傍8メッシュの最新週合計
    last = cw[cw.w == last_w][["row", "col", "count_w"]].rename(columns={"count_w": "c"})
    nbr = pd.Series(0.0, index=cells.index)
    key = cells.copy()
    for dr, dc in NEIGHBOR8:
        src = last.copy()
        src["row"] -= dr
        src["col"] -= dc
        merged = key.merge(src, on=["row", "col"], how="left")
        nbr = nbr.to_numpy() + merged["c"].fillna(0).to_numpy()
        nbr = pd.Series(nbr, index=cells.index)
    # weeks_since_last（T 時点）
    occ = cw[cw.count_w >= 1].groupby(["row", "col"])["w"].max()
    wsl = (T - occ).clip(upper=52)
    # yoy_lag: 前年同週 ±2週（T-54..T-50 = last_w-53..last_w-49）の件数
    yoy = (cw[cw.w.between(last_w - 53, last_w - 49)]
           .groupby(["row", "col"])["count_w"].sum())
    # mesh_rate: 当該メッシュの過去出没率（出没週数 / 総週数, 全期間=T より前）
    n_weeks = last_w + 1
    occ_weeks = (cw.assign(o=(cw["count_w"] >= 1).astype(int))
                 .groupby(["row", "col"])["o"].sum())

    f = cells.copy()
    idx = pd.MultiIndex.from_frame(cells)
    f["lag1_count"] = lag1.reindex(idx).fillna(0).to_numpy()
    f["roll4_count"] = roll4.reindex(idx).fillna(0).to_numpy()
    f["nbr8_lag1"] = nbr.to_numpy()
    f["weeks_since_last"] = wsl.reindex(idx).fillna(52).to_numpy()
    f["yoy_lag"] = yoy.reindex(idx).fillna(0).to_numpy()
    f["mesh_rate"] = (occ_weeks.reindex(idx).fillna(0).to_numpy() / n_weeks)
    f["month"] = target_date.month
    f["woy"] = int(target_date.isocalendar().week)
    return f, target_date


_ZIP = re.compile(r"〒?\s*\d{3}-?\d{4}")
_DIGIT = re.compile(r"[0-9０-９]")


def locality(addr: str, city: str) -> str:
    """地番情報から市町村より細かい地名（大字・町名）を抽出する。

    例: "秋田市添川字湯ノ岱11-2" → "添川"  /  "潟上市天王字…" → "天王"
    抽出できなければ空文字。
    """
    if not isinstance(addr, str) or not addr.strip():
        return ""
    s = _ZIP.sub("", addr).strip()
    if city and city in s:
        s = s.split(city, 1)[1]
    s = re.sub(r"^(大字|字)", "", s)
    # 数字（番地・丁目）以降を落とす。空白も区切りとして扱う
    m = re.search(r"[0-9０-９\s　]", s)
    if m:
        s = s[: m.start()]
    # 丁目・番地・地割マーカで切る（"字" は地名に含まれる場合があるので切らない）
    s = re.sub(r"(丁目|番地?|地割).*$", "", s)
    s = s.strip("　 ・,、-")
    return s[:7]


def _mode(s: pd.Series, default: str = "") -> str:
    vc = s.dropna().value_counts()
    return vc.index[0] if len(vc) else default


def enrich_area(f: pd.DataFrame, points: pd.DataFrame):
    """各メッシュに中心座標・市町村・地区（市町村+大字）・時間帯を付与。"""
    pts = assign_mesh(points.dropna(subset=["lat", "lon"]).copy())
    addr_col = "address" if "address" in pts.columns else None
    loc = ([locality(a, c) for a, c in zip(pts[addr_col], pts["city"])]
           if addr_col else [""] * len(pts))
    pts = pts.copy()
    pts["area"] = [f"{c}{l}" if l else (c or "") for c, l in zip(pts["city"], loc)]

    # (row,col) → 最頻 市町村 / 地区
    agg = (pts.groupby(["row", "col"])
           .agg(city=("city", _mode), area=("area", _mode)).reset_index())
    f = f.merge(agg, on=["row", "col"], how="left")
    f["city"] = f["city"].fillna("（周辺）")
    f["area"] = f["area"].fillna("（周辺）").replace("", "（周辺）")
    # メッシュ中心座標
    f["lat"] = (f["row"] + 0.5) * LAT_UNIT
    f["lon"] = (f["col"] + 0.5) * LON_UNIT

    # 時間帯（市町村別。地区は粗くなりがちなので市町村粒度で算出）
    pts["hour"] = pts["dt"].dt.hour
    pts["band"] = pts["hour"].map(time_band)
    city_band = pts.groupby("city")["band"].agg(lambda s: _mode(s, "—"))
    return f, city_band, pts


# ---------- ランキング出力 ----------

def rank_and_report(clf, f, city_band, pts, target_date):
    f = f.copy()
    f["prob"] = clf.predict_proba(f[FEATURES])[:, 1]
    f["band"] = f["city"].map(city_band).fillna("—")
    # 地区 → 市町村（時間帯ルックアップ用）
    area_to_city = f.groupby("area")["city"].agg(_mode)

    print(f"\n=== 次の週（{target_date:%Y-%m-%d} 週・最新データの翌週）出没しやすいエリア ===")
    print(f"\n--- メッシュ単位 Top {TOP_MESH}（≒1km） ---")
    print(f"{'順':>2} {'確率':>6}  {'地区':<16}{'時間帯':<12}{'直近4週':>6}  座標")
    top = f.sort_values("prob", ascending=False).head(TOP_MESH)
    for i, r in enumerate(top.itertuples(index=False), 1):
        print(f"{i:>2} {r.prob:>6.1%}  {r.area:<16}{r.band:<12}{int(r.roll4_count):>6}  "
              f"{r.lat:.3f},{r.lon:.3f}")

    print(f"\n--- 地区単位 Top {TOP_CITY}（期待出没メッシュ数 = 確率合計）---")
    area_rank = (f[f["area"] != "（周辺）"].groupby("area")
                 .agg(expected=("prob", "sum"), max_prob=("prob", "max"), n=("prob", "size"))
                 .sort_values("expected", ascending=False).head(TOP_CITY))
    print(f"{'順':>2} {'期待数':>7} {'最大確率':>7}  {'時間帯':<12}地区")
    for i, (area, r) in enumerate(area_rank.iterrows(), 1):
        band = city_band.get(area_to_city.get(area, ""), "—")
        print(f"{i:>2} {r.expected:>7.1f} {r.max_prob:>7.1%}  {band:<12}{area}")

    # 全体の時間帯分布
    print("\n--- 過去の出没 時間帯分布（全県）---")
    band_order = [b[2] for b in TIME_BANDS] + ["深夜(23-4)"]
    vc = pts["band"].value_counts()
    for b in band_order:
        n = int(vc.get(b, 0))
        print(f"  {b:<12}{n:>6,}  {'█' * int(n / 200)}")

    return f, top, area_rank, area_to_city


RISK_SURFACE_N = 400  # 地図のリスク面に出す上位メッシュ数
TREND_WEEKS = 26      # 地域トレンドの表示週数


def city_trends(full, panel, f):
    """市町村ごとの「モデル化リスク（期待出没メッシュ数）」の週次推移と、
    次週予測・前週比・前年同期比を計算する。"""
    p = panel.copy()
    p["prob"] = full.predict_proba(p[FEATURES])[:, 1]
    cmap = f[["row", "col", "city"]].drop_duplicates()
    p = p.merge(cmap, on=["row", "col"], how="left")
    p = p[p["city"].notna() & (p["city"] != "（周辺）")]
    weekly = p.groupby(["city", "week"])["prob"].sum().reset_index(name="risk")
    weeks_sorted = sorted(weekly["week"].unique())
    forecast = f[f["city"] != "（周辺）"].groupby("city")["prob"].sum()

    trends = {}
    for city, g in weekly.groupby("city"):
        s = g.set_index("week")["risk"].reindex(weeks_sorted).fillna(0.0)
        recent = s.iloc[-TREND_WEEKS:]
        cur = float(forecast.get(city, 0.0))
        last = float(s.iloc[-1])
        year_ago = float(s.iloc[max(0, len(s) - 1 - 52)])
        trends[city] = {
            "series": [{"w": w.strftime("%-m/%-d"), "risk": round(float(r), 2)}
                       for w, r in recent.items()],
            "forecast": round(cur, 2),
            "last": round(last, 2),
            "wow_pct": (round((cur / last - 1) * 100) if last > 0 else None),
            "yoy_pct": (round((cur / year_ago - 1) * 100) if year_ago > 0 else None),
            "areas": [],
        }

    # 市町村ごとの「地区」ランキング（選択した市町村で絞り込み表示するため）
    area_df = (f[f["area"] != "（周辺）"].groupby(["city", "area"])["prob"].sum()
               .reset_index(name="expected"))
    for city, sub in area_df.groupby("city"):
        if city not in trends:
            continue
        top = sub.sort_values("expected", ascending=False).head(8)
        trends[city]["areas"] = [{"area": a, "expected": round(float(e), 2)}
                                 for a, e in zip(top["area"], top["expected"])]
    return trends


def export_web(f, top, area_rank, area_to_city, city_band, pts, target_date, trends):
    """予測 Top メッシュ・地区ランキング・リスク面・地域トレンドを出力。"""
    band_order = [b[2] for b in TIME_BANDS] + ["深夜(23-4)"]
    vc = pts["band"].value_counts()
    risk = f.sort_values("prob", ascending=False).head(RISK_SURFACE_N)
    payload = {
        "target_week": target_date.strftime("%Y-%m-%d"),
        "trends": trends,
        "mesh_risk": [
            {"lat": round(float(r.lat), 5), "lon": round(float(r.lon), 5),
             "prob": round(float(r.prob), 4)}
            for r in risk.itertuples(index=False)
        ],
        "mesh_top": [
            {"lat": round(float(r.lat), 5), "lon": round(float(r.lon), 5),
             "prob": round(float(r.prob), 4), "city": r.area, "band": r.band}
            for r in top.itertuples(index=False)
        ],
        # key は web 互換のため city_top のまま。中身は地区（市町村+大字）粒度
        "city_top": [
            {"city": area, "expected": round(float(r.expected), 2),
             "max_prob": round(float(r.max_prob), 4),
             "band": city_band.get(area_to_city.get(area, ""), "—")}
            for area, r in area_rank.iterrows()
        ],
        "time_dist": [{"band": b, "n": int(vc.get(b, 0))} for b in band_order],
    }
    config.WEB_DIR.mkdir(parents=True, exist_ok=True)
    PRED_JS_PATH.write_text(
        "// 自動生成（src/train.py）。手で編集しない。\n"
        f"const KUMA_PRED = {json.dumps(payload, ensure_ascii=False)};\n",
        encoding="utf-8",
    )
    print(f"\n✓ 予測データ生成: {PRED_JS_PATH}")


def main() -> None:
    if not PANEL_PATH.exists():
        print("⚠ data/panel.parquet がありません。先に `python -m src.build_panel` を実行してください。")
        return
    panel = pd.read_parquet(PANEL_PATH)
    points = load_points()

    clf = train_and_eval(panel)
    # ランキングは全データで再学習したモデルで（最新の情報を反映）
    full = make_model()
    full.fit(panel[FEATURES], panel["y"], sample_weight=pos_weight(panel["y"]))

    f, target_date = next_week_features(panel)
    f, city_band, pts = enrich_area(f, points)
    f, top, area_rank, area_to_city = rank_and_report(full, f, city_band, pts, target_date)
    trends = city_trends(full, panel, f)
    export_web(f, top, area_rank, area_to_city, city_band, pts, target_date, trends)
    print("\n✓ M3 完了。web で『予測』レイヤを開くとランキングが見られます。")


if __name__ == "__main__":
    main()
