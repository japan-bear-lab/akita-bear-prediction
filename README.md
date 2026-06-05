# クマ出没予測 Web アプリ

秋田県のツキノワグマ出没オープンデータを使い、*メッシュ × 週次* でクマ出没を予測する Web アプリ。
"それっぽく動く" ではなく、時系列評価に耐える実用的な統計予測を目指す。

> *個人の趣味プロジェクトです。* 作者の所属組織とは一切関係ありません。
> 予測は統計的推定であり実際の出没を保証しません。安全判断は自治体・秋田県等の公式情報を優先してください。

設計の全体方針は別セッションの設計ドキュメントを参照（リポジトリには含めていません）。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # M1 は requests / pandas のみ
```

## M1: 取得 → 可視化

```bash
# CKAN 取得 → クレンジング → SQLite 保存 → web/points.js 生成 → サマリ表示
python -m src.build_db

# 地図を開く（出没点プロット）
open web/index.html
```

### スマホで表示確認する

同じ Wi-Fi 上の Mac で簡易サーバを立て、スマホのブラウザから Mac の LAN IP で開く:

```bash
python serve.py           # http://<MacのLAN-IP>:8000/ が表示される
```

表示された `http://192.168.x.x:8000/` をスマホ（同じ Wi-Fi）のブラウザで開く。
2 万点規模でも軽く動くよう地図は Canvas 描画。停止は Ctrl+C。
※ Mac の初回起動時、ファイアウォールの着信許可ダイアログが出たら「許可」。

`src/build_db.py` が以下を実行する:

1. CKAN（秋田県クマダス）から JSON 取得・クレンジング（`src/data_loader.py`）
2. `data/kumadas.db`（SQLite）に保存
3. 件数・期間・市町村別件数・情報種別内訳を標準出力
4. `web/points.js` を生成（Leaflet が読む出没点。`file://` でも CORS に阻まれない）

## M2: メッシュ × 週パネル化

```bash
# 出没点（SQLite or CKAN）→ (メッシュ×週) パネル + リークフリー特徴量
# → data/panel.parquet 保存 → 時系列分割でベースライン評価
python -m src.build_panel
```

`src/panel.py` が以下を行う:

- jismesh で各出没点に 3 次メッシュ（≒1km）を付与。隣接計算用に整数グリッド `(row,col)` も保持
- 2022 年以降（収録カバレッジ交絡を除外）× 「出没メッシュ + 8 近傍バッファ」に対象を限定
- (メッシュ × 月曜起点週) の全格子を作り `y = その週に出没≥1`
- *リークフリー特徴量*（すべて t より前のみ）: 前週件数 / 過去4週合計 / 近傍8メッシュ前週合計 / 最終出没からの週数 / 月 / 週番号

## M3: 翌週出没確率モデル + リスクランキング

```bash
# パネルで GBDT 学習 → 時系列分割で評価 → 「次の週」エリア別出没確率を予測
# → ランキング表示 + web/predictions.js（地図リスク面）を生成
python -m src.train
```

`src/train.py`:

- モデルは sklearn `HistGradientBoostingClassifier`（LightGBM 同等の GBDT・`libomp` 不要）
- 時系列分割（過去で学習 → 未来で検証）で ROC-AUC / PR-AUC を M2 ベースラインと比較
- 最新週の「次の週」について各メッシュの出没確率を予測 → *メッシュ単位 / 市町村単位*でランキング
- 過去の目撃時刻から*出没しやすい時間帯*を算出（※通報バイアス注意）

### 可視化（地図ダッシュボード）

`web/index.html` は 2 モードのインタラクティブ地図:

- *次週リスク*: 予測確率を逐次配色（YlOrRd）のリスク面で表示 + 市町村ランキング（横棒）+ 時間帯チャート + 最高リスク地点のパルス表示
- *出没実績*: 全出没点（公式/市民投稿の色分け）

```bash
python serve.py     # スマホ（同じ Wi-Fi）でも http://<MacのLAN-IP>:8000/ で見られる
```

## 実行順（フルパイプライン）

```bash
python -m src.build_db      # M1: 取得・クレンジング・SQLite・出没点
python -m src.build_panel   # M2: メッシュ×週パネル + 特徴量
python -m src.train         # M3: 学習・予測・ランキング・リスク面
python serve.py             # 地図ダッシュボードを配信
```

## マイルストーン

- *M1* ✅ CKAN 取得 → クレンジング → Leaflet で出没点プロット
- *M2* ✅ 1km メッシュ集計 → 週次パネル化 → リークフリー特徴量 + 時系列分割ベースライン
- *M3* ✅ GBDT → 翌週確率予測 → 時系列分割で AUC / PR-AUC 評価 → リスクランキング + 地図
- *M4* 外部特徴量（堅果類豊凶・土地被覆・標高・気象）で精度改善

## データ更新（月次）

クマダスは毎月新しいリソースが発行される可能性がある。最新の `resource_id` を
[データセットページ](https://ckan.pref.akita.lg.jp/dataset/050008_shizenhogoka_003)
で確認し、`src/config.py` の `RESOURCE_ID` を差し替えるだけで追従できる。

## GitHub Pages へデプロイ（Org `japan-bear-lab` / public リポジトリ）

URL に個人名を出さないため、中立名の Organization 配下に置く。
公開 URL: `https://japan-bear-lab.github.io/akita-bear-prediction/`

`.github/workflows/deploy.yml` が *データ取得 → 学習 → Pages デプロイ* を自動化する
（push 時・毎月 1 日・手動実行）。

```bash
# 1. Organization を作成（無料）: ブラウザで
#    https://github.com/organizations/plan → Free → 名前 "japan-bear-lab"

# 2. gh が個人アカウント(org のオーナー)で有効か確認
gh auth status
# gh auth switch --user ryosei-hatakeyama   # 必要なら

# 3. 初回コミット
git add -A && git commit -m "init: クマ出没予測アプリ"

# 4. org 配下に public リポジトリを作成して push
gh repo create japan-bear-lab/akita-bear-prediction --public --source=. --remote=origin --push

# 5. Pages のソースを「GitHub Actions」に設定
gh api -X POST repos/japan-bear-lab/akita-bear-prediction/pages -f build_type=workflow

# 6. Actions の進行を確認（数分で完了）
gh run watch
```

以降は `git push` または毎月の cron で自動更新。`web/index.html` は相対パス参照なので
サブパス配信（`/akita-bear-prediction/`）でもそのまま動く。

> 無料の GitHub Pages は *public リポジトリ必須*（private は Pro）。
> 本データは公開オープンデータ・本コードは個人プロジェクトのため public で問題なし。
> Org 配下にすると URL ホストが `japan-bear-lab` になり、個人名が URL に出ない。

## SEO / シェア最適化

`web/index.html` に以下を実装済み:

- *title / meta description*（キーワード: 秋田県・ツキノワグマ・クマ出没・予測・マップ）
- *Open Graph + Twitter Card*（SNS リンクプレビュー）。画像は `web/ogp.svg` を CI で `ogp.png`(1200×630) に変換
- *構造化データ (JSON-LD)* — `WebApplication` + データ出典 `Dataset`(CC BY)
- *canonical / theme-color / SVG favicon / `<noscript>` フォールバック*
- `web/robots.txt` + `web/sitemap.xml`

> ⚠️ *公開 URL*: `https://japan-bear-lab.github.io/akita-bear-prediction/` を `index.html`（canonical・og:url・og:image・twitter:image・JSON-LD）/
> `robots.txt` / `sitemap.xml` に埋め込み済み。配置先（org / repo 名）を変える場合はこれらの URL を置換すること。
> 公開後は [Google Search Console](https://search.google.com/search-console) にサイト登録 + sitemap 送信で検索インデックスを促進できる。

## 公開する場合のセキュリティ

このアプリは *バックエンドなし・ユーザー入力の永続化なし・公開オープンデータのみ* なので攻撃面は小さいが、公開時は以下を守る:

- *`serve.py` を公開サーバにしない*。これは LAN 開発用（`python http.server` ベース・HTTPS なし・無認証）。公開は *静的ホスティング*（GitHub Pages / Cloudflare Pages / Netlify 等）に `web/`（+ 生成された `points.js` / `predictions.js`）を置く。HTTPS と CDN が付き、守るべきサーバが無くなる。
- *XSS 対策済み*: 地名・`目撃時の状況` は市民投稿（真偽不明）由来の自由記述を含むため、画面に挿入する全データを HTML エスケープ（`esc()`）している。
- *CSP 設定済み*: `index.html` に Content-Security-Policy を付与（script/style/img/font の取得元を自分自身 + 使用 CDN に限定）。
- *SRI*: Leaflet の CDN 読み込みに integrity 属性を付与済み。
- *免責表示*: 「統計的推定であり実際の出没を保証しない。安全判断は自治体・公式情報を優先」を UI に明記。
- *秘密情報なし*: API キー等は不要（CartoDB dark ベースマップは鍵不要）。`data/` は `.gitignore` 済み。

## ライセンス・出典

- 出没データ: *出典「秋田県クマダス（ツキノワグマ等情報マップシステム）」* / Creative Commons Attribution 4.0 (CC BY 4.0)
- 地図: © OpenStreetMap contributors
- 市民投稿データ（`【投稿】` 接頭辞）は真偽不明の注記がある旨、予測結果には免責表示を行う
