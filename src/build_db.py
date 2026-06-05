"""M1 のメイン処理。

1. CKAN から取得・クレンジング
2. SQLite (data/kumadas.db) に保存
3. 件数・期間・市町村別件数を標準出力
4. Leaflet 用に web/points.js を生成（ブラウザで index.html を直接開ける）

実行: python -m src.build_db
"""

from __future__ import annotations

import json
import sqlite3

import pandas as pd

from . import config
from .data_loader import load_kumadas

# SQLite に永続化する列（解析の素地）
_DB_COLUMNS = [
    "id", "dt", "city", "address", "kind", "species",
    "lat", "lon", "sex", "group_type", "count", "note", "is_citizen_post",
]


def save_sqlite(df: pd.DataFrame) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = df.reindex(columns=[c for c in _DB_COLUMNS if c in df.columns]).copy()
    # SQLite は datetime を持てないので ISO 文字列に
    out["dt"] = out["dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(config.DB_PATH) as conn:
        out.to_sql("kumadas", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kumadas_dt ON kumadas(dt)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kumadas_city ON kumadas(city)")
    print(f"✓ SQLite 保存: {config.DB_PATH}  ({len(out)} 行)")


def write_points_js(df: pd.DataFrame) -> None:
    """Leaflet が読む出没点を JS 変数として書き出す。

    file:// で index.html を開いても CORS に阻まれないよう、GeoJSON を
    fetch せず <script> で直接読み込ませる方式にしている。
    """
    points = [
        {
            "lat": round(float(r.lat), 5),
            "lon": round(float(r.lon), 5),
            "dt": r.dt.strftime("%Y-%m-%d"),
            "city": r.city if isinstance(r.city, str) else "",
            "kind": r.kind if isinstance(r.kind, str) else "",
            "citizen": bool(r.is_citizen_post),
        }
        for r in df.itertuples(index=False)
    ]
    config.WEB_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "count": len(points),
        "from": df["dt"].min().strftime("%Y-%m-%d"),
        "to": df["dt"].max().strftime("%Y-%m-%d"),
        "attribution": config.ATTRIBUTION,
    }
    body = (
        "// 自動生成ファイル（src/build_db.py）。手で編集しない。\n"
        f"const KUMA_META = {json.dumps(meta, ensure_ascii=False)};\n"
        f"const KUMA_POINTS = {json.dumps(points, ensure_ascii=False)};\n"
    )
    config.POINTS_JS_PATH.write_text(body, encoding="utf-8")
    print(f"✓ Leaflet データ生成: {config.POINTS_JS_PATH}  ({len(points)} 点)")


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== クマダス取得サマリ（ツキノワグマ） ===")
    print(f"総件数        : {len(df):,}")
    print(f"期間          : {df['dt'].min():%Y-%m-%d} 〜 {df['dt'].max():%Y-%m-%d}")
    citizen = int(df["is_citizen_post"].sum())
    print(f"市民投稿(【投稿】): {citizen:,} 件 ({citizen / len(df):.1%})")

    print("\n--- 市町村別 件数 Top 15 ---")
    by_city = df["city"].value_counts().head(15)
    for city, n in by_city.items():
        print(f"  {city:<10} {n:>5,}")

    if "kind" in df.columns:
        print("\n--- 情報種別 内訳 ---")
        for kind, n in df["kind"].value_counts().items():
            print(f"  {kind:<14} {n:>5,}")


def main() -> None:
    print("CKAN から取得中 ...")
    df = load_kumadas()
    if df.empty:
        print("⚠ クレンジング後に 0 件。resource_id / スキーマを確認してください。")
        return
    print_summary(df)
    save_sqlite(df)
    write_points_js(df)
    print("\n✓ M1 完了。web/index.html をブラウザで開くと出没点が表示されます。")


if __name__ == "__main__":
    main()
