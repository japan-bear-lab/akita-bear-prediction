"""CKAN（秋田県クマダス）から出没データを取得し、クレンジングする。

ドキュメント 8 章の参考実装をベースに、堅牢性（JSON 構造ゆらぎ・空キー列・
変則命名）を補強している。
"""

from __future__ import annotations

import json

import pandas as pd
import requests

from . import config


def fetch_raw(url: str = config.DOWNLOAD_URL, timeout: int = 60) -> list[dict]:
    """CKAN から生 JSON を取得。生データは data/ にキャッシュする。"""
    res = requests.get(url, timeout=timeout)
    res.raise_for_status()
    payload = res.json()

    # CKAN のリソース DL は「レコードの配列」がそのまま返るのが基本だが、
    # {"result": {"records": [...]}} のような datastore_search 形式も吸収する。
    if isinstance(payload, dict):
        if "records" in payload:
            payload = payload["records"]
        elif "result" in payload and isinstance(payload["result"], dict):
            payload = payload["result"].get("records", [])

    if not isinstance(payload, list):
        raise ValueError(f"想定外の JSON 構造: top-level が {type(payload).__name__}")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.RAW_JSON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


# 日本語 → 英語の列名対応（x(緯度)=lat / y(経度)=lon の変則命名に注意）
_RENAME = {
    "x(緯度)": "lat",
    "y(経度)": "lon",
    "目撃日時": "dt",
    "情報種別": "kind",
    "獣種": "species",
    "市町村": "city",
    "出没情報ID": "id",
    "目撃時の状況": "note",
    "地番情報": "address",
    "性別": "sex",
    "単独か親子": "group_type",
    "頭数": "count",
}


def cleanse(records: list[dict]) -> pd.DataFrame:
    """生レコードを予測テーブルの素地となる DataFrame に整形する。"""
    df = pd.json_normalize(records)

    # 末尾の空キー列 "" を落とす
    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]

    df = df.rename(columns=_RENAME)

    # 型変換
    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
    df["count"] = pd.to_numeric(df.get("count"), errors="coerce")
    # "2026/4/30 19:50" 形式。曖昧さ回避のため format を明示せず coerce。
    df["dt"] = pd.to_datetime(df.get("dt"), errors="coerce")

    # 市民投稿フラグ（除外せず信頼度特徴量として残す）
    df["is_citizen_post"] = df.get("note", "").fillna("").str.startswith("【投稿】")

    # 緯度経度・日時が欠損する行は予測に使えないので落とす
    df = df.dropna(subset=["lat", "lon", "dt"])

    # ツキノワグマに絞る
    if "species" in df.columns:
        df = df[df["species"] == config.TARGET_SPECIES].copy()

    # ID で重複排除
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])

    # 秋田県の妥当な座標範囲でガード（明らかな異常値除去）
    df = df[(df["lat"].between(38.5, 41.5)) & (df["lon"].between(139.0, 141.5))]

    return df.sort_values("dt").reset_index(drop=True)


def load_kumadas() -> pd.DataFrame:
    """取得 → クレンジングの一括実行。"""
    return cleanse(fetch_raw())


if __name__ == "__main__":
    df = load_kumadas()
    print(df.shape)
    print(df[["dt", "city", "lat", "lon", "kind"]].head())
