# Generated Index Usage Examples

この文書は、`generate-RAG-from-jsonl` で生成した `index/` 一式を、実際にどのように利用するかを説明します。

`README.md` は主に「RAGを生成する方法」を説明します。
この文書はその続きとして、**生成後のFAISS / SQLite / vectorデータをどう扱うか**に絞っています。

---

## 1. 生成された index は「一式」で使う

production buildが完了すると、出力先の `index/` にはおおむね次のファイルが生成されます。

```text
index/
├─ metadata.sqlite
├─ trained_template.faiss
├─ config.json
├─ progress.json
├─ build.log
│
├─ shard_000000.faiss
├─ shard_000000.f16
├─ shard_000000.ids
├─ shard_000000.json
│
├─ shard_000001.faiss
├─ shard_000001.f16
├─ shard_000001.ids
├─ shard_000001.json
│
└─ ...
```

これらは個別に使うというより、**`index/` ディレクトリ全体で1つのRAGデータセット**と考えます。

利用側では、

```text
config.json
metadata.sqlite
shard_*.json
shard_*.faiss
必要に応じて shard_*.f16 / shard_*.ids
```

を相互に対応させて使用します。

したがって、別のデータから新しいRAGを生成した場合も、基本的には **indexフォルダ一式を差し替える** という扱いになります。

---

# 2. 各ファイルの役割

## `config.json`

RAG生成時の設定を記録します。

主に、

```text
Embeddingモデル
Embedding次元数
query/document prefix
chunk設定
FAISS設定
nprobe
candidate_count
float16 rerankの利用有無
```

などが保存されます。

利用側はこのファイルを読み、**RAG生成時と同じEmbedding条件で質問をEmbeddingする**必要があります。

そのため、利用側でEmbeddingモデル名や次元数を別途ハードコードするより、`config.json` を基準にする方が安全です。

---

## `metadata.sqlite`

chunk本文と、検索・表示に必要なmetadataを格納するSQLiteデータベースです。

現在の構成では、たとえば次のような情報が検索側から利用されます。

```text
chunk_id
article_id
title
url
section
chunk_no
chunk_count
text
page_type
quality_weight
vector_shard
vector_row
normalized_title
```

また、タイトル検索用のindex / FTSも含まれます。

役割としては、

```text
FAISS
    → どのchunkが近いか

metadata.sqlite
    → そのchunkの本文・タイトル・文書情報は何か
```

という関係です。

FAISSだけではRAG回答に渡す本文を復元できないため、`metadata.sqlite` も必要です。

---

## `shard_XXXXXX.faiss`

FAISSの検索indexです。

質問をEmbeddingしたvectorを使って、

```text
どのchunk vectorが近いか
```

を高速に検索します。

データ量が大きくても扱いやすいよう、現在は複数shardに分割して保存します。

---

## `shard_XXXXXX.json`

各shardのmanifestです。

主に、

```text
shard番号
記事数
chunk数
Embedding次元数
対応する .faiss ファイル
対応する .f16 ファイル
対応する .ids ファイル
```

などを記録します。

利用側はこのmanifestを読んで、どのファイルが1セットなのかを判断できます。

---

## `shard_XXXXXX.f16`

`save_float16=true` の場合に生成されます。

FAISSへ格納した近似indexとは別に、Embedding vectorをfloat16で保存したものです。

現在の検索方式では、

```text
FAISSで候補を取る
    ↓
.f16 のvectorで類似度を再計算
```

というrerankに利用できます。

FAISSの近似圧縮による誤差を、後段である程度補うためのデータです。

---

## `shard_XXXXXX.ids`

`.f16` とchunk IDの対応関係を保持します。

`.f16` を利用する構成では、対応する `.ids` もセットで扱います。

---

## `trained_template.faiss`

production shardを作成するためにtrainingされたFAISS templateです。

主に **生成工程で再利用するためのファイル** です。

通常の検索利用では各 `shard_*.faiss` を使用するため、利用側がこのファイルを直接検索する必要はありません。

ただし、同じindex設定でbuildを継続・再生成する場合には有用です。

---

## `progress.json`

production buildの進捗状態です。

```text
現在のshard
完了済みshard
入力JSONL上の位置
build状態
```

などを記録します。

検索時には基本的に使用しません。

中断したproduction buildを再開するときの管理情報です。

---

## `build.log`

RAG生成時のログです。

検索時には使用しません。

buildに問題があった場合や、Embedding短縮・エラー・処理時間などを確認するときに使います。

---

# 3. LocalLLM_with_Wikipedia で使う

現在の実装では、`LocalLLM_with_Wikipedia` が `index/` 一式を直接読み込めます。

想定配置:

```text
RAG/
├─ generate-RAG-from-jsonl/
├─ LocalLLM_with_Wikipedia/
└─ data/
   └─ index/
      ├─ config.json
      ├─ metadata.sqlite
      ├─ shard_000000.faiss
      ├─ shard_000000.json
      ├─ shard_000000.f16
      ├─ shard_000000.ids
      └─ ...
```

`LocalLLM_with_Wikipedia` はデフォルトで、

```text
RAG/data/index/
```

を検索データとして使用します。

したがって、`generate-RAG-from-jsonl` のproduction buildを最初からこの場所へ出力している場合は、そのまま利用できます。

---

# 4. 別のデータで作ったRAGへ差し替える

たとえばWikipediaではなく、CSVから作った独自データを使う場合を考えます。

```text
my_data.csv
    ↓
JSONLへ変換
    ↓
generate-RAG-from-jsonl
    ↓
my_index/
```

JSONLが `generate-RAG-from-jsonl` の入力仕様を満たしていれば、RAG生成工程は同じです。

最低限の推奨形式:

```json
{
  "text": "本文",
  "meta": {
    "id": "一意な文書ID",
    "title": "文書タイトル",
    "url": ""
  }
}
```

※ `meta.url` は任意です。

生成された、

```text
my_index/
```

の内容を現在の、

```text
RAG/data/index/
```

と差し替えれば、`LocalLLM_with_Wikipedia` の検索エンジンは新しいRAGを読み込みます。

概念的には、

```text
CSV
 ↓
JSONL
 ↓
generate-RAG-from-jsonl
 ↓
index/
 ↓
LocalLLM_with_Wikipedia
```

という流れです。

利用側は、元データがCSVだったかWikipedia XMLだったかを知る必要はありません。

---

# 5. 安全にindexを切り替える例

既存のWikipedia indexを残しておきたい場合は、削除するより名前を変えて保管する方が安全です。

例:

```text
data/
├─ index_wikipedia/
├─ index_manuals/
└─ index/
```

使いたいものを `index/` にする方法でも構いません。

あるいは `LocalLLM_with_Wikipedia` の `--index` オプションを使えば、フォルダ名を変更せず直接指定できます。

例:

```cmd
python webui.py --index ..\data\index_manuals
```

これなら複数RAGを同時に保管できます。

```text
data/
├─ index_wikipedia/
├─ index_company_docs/
├─ index_manuals/
└─ index_test/
```

必要に応じて起動時に切り替えます。

---

# 6. 独自データでも現在の検索方式を使える条件

`LocalLLM_with_Wikipedia` の検索エンジンは、単純なvector searchだけではなく、

```text
文書タイトルを特定
    ↓
その文書内のchunkを検索
```

というarticle-focused searchを持っています。

そのため、Wikipedia以外でも、

```text
1 JSONL record = 1文書
meta.id = 文書の一意なID
meta.title = 文書名
text = 文書本文
```

という構造のデータとは相性がよいです。

たとえば、

```text
製品マニュアル
技術文書
FAQ集
規定集
社内手順書
ナレッジベース
```

のように、各文書に明確なタイトルがあるデータなら、現在の検索方式を比較的そのまま利用できます。

---

# 7. 現在の検索方式がWikipedia向けに最適化されている部分

`generate-RAG-from-jsonl` 自体はJSONLからRAGを作る処理として比較的汎用的ですが、現在の `LocalLLM_with_Wikipedia` の検索エンジンにはWikipedia向けの調整があります。

代表的なもの:

```text
title exact match
title表記揺れ
article_focus
記事内再検索
タイトルFTS
ストーリー/物語等のintent補正
```

特に検索設計は、

```text
どの記事か
    ↓
記事内のどこか
```

を先に検索エンジンが決め、その結果をLLMへ渡す考え方です。

独自データでも「1文書に明確なtitleがある」場合には利用しやすい一方、

```text
titleを持たない大量の短文
ログ
時系列イベント
表形式の1行データ
```

などでは、検索方式をカスタマイズした方がよい場合があります。

---

# 8. Search modeの使い分け例

生成したRAGを `LocalLLM_with_Wikipedia` で使う場合、現在は複数の検索モードがあります。

## `article_focus`

特定文書について質問する場合に向きます。

例:

```text
製品Aマニュアルのバックアップ手順を教えて
```

`製品Aマニュアル` がtitleとして存在すれば、その文書内を重点的に検索できます。

---

## `auto`

タイトル一致と全体vector searchを自動的に組み合わせます。

通常利用向けです。

---

## `strict`

titleが確認できる文書だけを検索したい場合に向きます。

似ている別文書を勝手に採用したくない用途で有効です。

---

## `balanced`

本命文書を中心にしつつ、関連する別文書も参照します。

---

## `discovery`

文書タイトルを特定せず、複数文書を横断して意味検索したい場合に向きます。

独自データを初めて試す場合、

```text
article_focus
auto
discovery
```

あたりを比較すると、そのデータにどの検索方式が合うか判断しやすくなります。

---

# 9. Article Viewerについて

`LocalLLM_with_Wikipedia` に含まれる現在のArticle Viewer、

```text
JSONL Viewer
Kiwix
```

はWikipediaの元記事を直接確認するための機能です。

これは、

```text
Wikipediaの1記事全体はLLMへ直接渡すには大きい
    ↓
RAGではchunkを使う
    ↓
人間が原文を確認するときは元記事をViewerで開く
```

という目的で追加されています。

そのため、CSVなど別データから作ったRAGへindexを差し替えた場合、

```text
RAG検索
LLMへのコンテキスト供給
回答生成
```

は利用できますが、Wikipedia用Article Viewerがそのまま独自データを表示できるわけではありません。

独自データにも原文Viewerが必要なら、そのデータ用のViewer backendを別途用意します。

Article Viewerインターフェースは概ね、

```text
open(title)
```

という単純な構造なので、用途に合わせたViewerを追加する余地があります。

---

# 10. 別プログラムから直接indexを使う場合

`LocalLLM_with_Wikipedia` を使わず、自分のPythonプログラムから生成済みindexを利用することもできます。

基本的な処理は、

```text
config.jsonを読む
 ↓
同じEmbedding設定でqueryをvector化
 ↓
shard_*.faissを検索
 ↓
chunk_idを得る
 ↓
metadata.sqliteから本文を取得
 ↓
必要なら .f16 でrerank
 ↓
LLMへ渡す
```

です。

簡略化すると、

```text
query
  │
  ▼
Embedding
  │
  ▼
FAISS
  │ chunk_id
  ▼
SQLite
  │ text / title / metadata
  ▼
LLM
```

となります。

現在の `LocalLLM_with_Wikipedia/wikirag/search_engine.py` は、この利用方法の実装例として参照できます。

---

# 11. 最低限必要なもの

単純な検索実装を自作する場合でも、最低限、

```text
config.json
metadata.sqlite
shard_*.json
shard_*.faiss
```

は必要です。

現在と同じfloat16 rerankも利用するなら、

```text
shard_*.f16
shard_*.ids
```

も保持します。

`progress.json` と `build.log` はbuild管理用なので、検索だけを行う環境では必須ではありません。

`trained_template.faiss` も通常の検索実行には必須ではありません。

---

# 12. まとめ

`generate-RAG-from-jsonl` の出力は、個々のファイルを独立して使うものではなく、

```text
index/
```

全体で1つのRAGデータセットです。

最も簡単な利用方法は、

```text
JSONLを用意
 ↓
generate-RAG-from-jsonl
 ↓
indexを生成
 ↓
LocalLLM_with_Wikipedia が参照する data/index に配置
 ↓
起動
```

です。

別のデータセットへ切り替える場合も、

```text
新しいJSONL
 ↓
新しいindex
 ↓
--index で指定
```

または `data/index` を差し替えるだけで利用できます。

このため `generate-RAG-from-jsonl` は、Wikipediaだけでなく、必要なJSONL形式へ変換できる文書データに対してRAG検索データを生成する用途にも利用できます。
