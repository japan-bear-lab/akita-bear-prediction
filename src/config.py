"""設定値。月次でデータセットが更新される可能性があるため、resource_id をここに切り出す。

毎月新しい resource が発行された場合は、データセットページ
https://ckan.pref.akita.lg.jp/dataset/050008_shizenhogoka_003
で最新の resource_id を確認し、RESOURCE_ID を差し替えるだけで追従できる。
"""

from pathlib import Path

# --- CKAN（秋田県オープンデータポータル / クマダス） ---
DATASET_ID = "050008_shizenhogoka_003"
# 20260430 時点の JSON リソース。毎月更新の可能性 → ここだけ直せば追従できる。
RESOURCE_ID = "1307fa97-8cbb-4726-9da7-12bd4aa4a220"

CKAN_BASE = "https://ckan.pref.akita.lg.jp"
DOWNLOAD_URL = (
    f"{CKAN_BASE}/dataset/{DATASET_ID}"
    f"/resource/{RESOURCE_ID}/download?user-download=true"
)

# 対象獣種（M1 はツキノワグマのみ）
TARGET_SPECIES = "ツキノワグマ"

# --- パス ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WEB_DIR = PROJECT_ROOT / "web"
DB_PATH = DATA_DIR / "kumadas.db"
RAW_JSON_PATH = DATA_DIR / "kumadas_raw.json"  # 取得した生 JSON のキャッシュ
POINTS_JS_PATH = WEB_DIR / "points.js"  # Leaflet が読む出没点（CORS 回避のため JS で埋め込み）

# 出典表記（CC BY 4.0）
ATTRIBUTION = "秋田県クマダス（ツキノワグマ等情報マップシステム）"
