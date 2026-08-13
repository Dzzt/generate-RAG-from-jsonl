# generate-RAG-from-jsonl 入力JSONL仕様

## 概要

`generate-RAG-from-jsonl`
は、**1行を1文書（Wikipediaの場合は1記事）とするJSON
Lines形式**のデータを入力として、文章をチャンク化し、Embedding、FAISS
index、検索用SQLiteなどのRAGデータを生成します。

現在の実装が実際にRAG生成へ使用する基本構造は、次の4項目です。

``` json
{
  "text": "本文...",
  "meta": {
    "id": "12345",
    "title": "記事タイトル",
    "url": "https://example.com/article/12345"
  }
}
```

Wikipedia用の `wikipediadump_xml-to-jsonl`
は、これにrevision情報などを加えた、より詳細な形式を出力します。

この文書では、

1.  `generate-RAG-from-jsonl` が実際に必要とする項目
2.  Wikipedia変換プロジェクトが出力する標準形式
3.  独自JSONLをRAG化する場合の注意点

を分けて説明します。

------------------------------------------------------------------------

# 1. ファイル形式

## JSON Lines（JSONL）

通常のJSON配列ではなく、**1行ごとに独立したJSON object**を記述します。

正しい例:

``` jsonl
{"text":"文書1の本文","meta":{"id":"1","title":"文書1","url":"https://example.com/1"}}
{"text":"文書2の本文","meta":{"id":"2","title":"文書2","url":"https://example.com/2"}}
{"text":"文書3の本文","meta":{"id":"3","title":"文書3","url":"https://example.com/3"}}
```

次のようなJSON配列ではありません。

``` json
[
  {"text":"文書1の本文","meta":{"id":"1","title":"文書1"}},
  {"text":"文書2の本文","meta":{"id":"2","title":"文書2"}}
]
```

`generate-RAG-from-jsonl` はファイルを1行ずつ読み、

``` python
obj = json.loads(line)
```

として処理します。

そのため、大きなJSONLでもファイル全体をメモリへ読み込む必要がありません。

------------------------------------------------------------------------

# 2. 文字コード

JSONLの内容は **UTF-8** を前提とします。

日本語本文・日本語タイトルをそのまま格納できます。

例:

``` json
{"text":"大谷翔平は、日本出身のプロ野球選手。","meta":{"id":"123","title":"大谷翔平","url":"https://ja.wikipedia.org/wiki/大谷翔平"}}
```

JSONとして正しくescapeされていれば、本文中に改行や引用符を含めることもできます。

実際のJSONLでは、1記事は物理的には1行なので、本文中の改行はJSON文字列中の
`\n` として記録されます。

------------------------------------------------------------------------

# 3. 圧縮形式

production buildは、

``` text
.jsonl
.jsonl.bz2
```

の両方を読み込めます。

`.bz2` の場合も、事前に展開して別ファイルを作る必要はありません。

例:

``` text
wikipedia_ja_from_dump.jsonl.bz2
```

をそのまま入力として使用できます。

現在のproduction builderは拡張子が `.bz2` ならPythonの `bz2`
moduleでストリーミング展開しながら読み込みます。

------------------------------------------------------------------------

# 4. 基本レコード

`generate-RAG-from-jsonl`
がRAG生成時に読み取る情報は、基本的に次の形です。

``` json
{
  "text": "本文",
  "meta": {
    "id": "文書ID",
    "title": "文書タイトル",
    "url": "元文書のURL"
  }
}
```

各項目を詳しく説明します。

------------------------------------------------------------------------

## `text`

### 役割

RAG化する本文です。

``` json
"text": "ここに文書本文を入れる"
```

この文章が `ChunkBuilder`
に渡され、設定された文字数を基準に複数のチャンクへ分割されます。

その後、各チャンクがEmbeddingされます。

### 実質的な重要度

RAGの内容そのものなので、最重要項目です。

現在の実装では `text` が存在しない場合、

``` text
""
```

として扱われるため、プログラム上は必須チェックされません。

しかし本文が空ではRAG文書として意味がないため、**入力仕様としては必須と考えるべき項目**です。

### 推奨

-   プレーンテキストにしておく
-   不要なHTMLやWiki markupは事前に除去する
-   段落区切りは残す
-   本文中の改行は保持してよい
-   1レコードに1つの論理的な文書を入れる

Wikipedia変換では、Wikitextをある程度クリーニングした文章を `text`
に格納しています。

------------------------------------------------------------------------

## `meta`

本文に付随するmetadataを格納するobjectです。

``` json
"meta": {
  "id": "12345",
  "title": "記事タイトル",
  "url": "https://..."
}
```

現在のRAG生成処理が直接使用するのは、

``` text
id
title
url
```

です。

`meta` 自体が存在しない場合も空objectとして処理されますが、特に `title`
は検索品質に重要なので、通常は必ず用意します。

------------------------------------------------------------------------

# 5. `meta.id`

文書を識別するIDです。

``` json
"id": "361160"
```

文字列でも数値でも構いません。RAG生成時には文字列へ変換されます。

### 推奨

**文書ごとに一意で、再生成しても変化しないID**を使用してください。

Wikipediaの場合はpage IDを使用しています。

### idがない場合

現在のproduction
builderは、IDが存在しない場合に入力位置をもとにした代替IDを自動生成します。

そのため技術的には省略可能です。

ただし、

-   RAGを作り直したときの追跡
-   metadataの確認
-   文書の同一性

を考えると、独自データでも安定したIDを持たせることを推奨します。

------------------------------------------------------------------------

# 6. `meta.title`

文書タイトルです。

``` json
"title": "サイバーパンク2077"
```

現在のWikipedia向け検索システムでは非常に重要な項目です。

`LocalLLM_with_Wikipedia` では、単純なvector similarityだけでなく、

``` text
質問
  ↓
どの記事か
  ↓
記事内のどこか
```

という検索を行うため、記事タイトルを積極的に利用します。

### titleがない場合

現在のbuilderは、

``` text
(無題)
```

を代入するため、RAG生成自体は可能です。

しかしWikipedia向け検索方式ではtitleを使った検索能力を失うため、**実用上は必須**です。

### 独自データの場合

Wikipedia以外でも、

``` text
製品名
文書名
マニュアルの章名
FAQの質問名
資料タイトル
```

など、その文書を人間が識別できる名称を入れると扱いやすくなります。

------------------------------------------------------------------------

# 7. `meta.url`

元文書を示すURLです。

``` json
"url": "https://ja.wikipedia.org/wiki/サイバーパンク2077"
```

RAG生成処理ではmetadataとして保存されます。

### URLがない場合

空文字列として扱われるため、省略可能です。

ローカル文書など、対応するWeb URLが存在しないデータなら、

``` json
"url": ""
```

でも構いません。

URL以外の識別子をここへ無理に入れる必要はありません。

------------------------------------------------------------------------

# 8. 最小推奨形式

独自JSONLを `generate-RAG-from-jsonl`
へ入力する場合、最低限、次の形式を推奨します。

``` jsonl
{"text":"文書1の本文です。","meta":{"id":"doc-0001","title":"文書1","url":""}}
{"text":"文書2の本文です。","meta":{"id":"doc-0002","title":"文書2","url":""}}
```

特に重要なのは、

``` text
text
meta.id
meta.title
```

です。

`url` は元文書へのリンクがある場合に設定します。

------------------------------------------------------------------------

# 9. Wikipedia用標準形式

`wikipediadump_xml-to-jsonl` が生成するWikipedia
JSONLは、現在次の形式です。

``` json
{
  "text": "記事本文...",
  "meta": {
    "id": "361160",
    "title": "ガイ・フォークス",
    "url": "https://ja.wikipedia.org/wiki/ガイ・フォークス",
    "revision_id": "107436472",
    "timestamp": "2025-12-01T03:27:24Z",
    "sha1": "l1nzn5bhoe8pcaiihssaoj8l5e9gjls",
    "redirect": false,
    "redirect_target": null,
    "source": "wikimedia-pages-articles",
    "source_dump": "jawiki-....xml.bz2"
  }
}
```

このうち、現在の `generate-RAG-from-jsonl` がRAG生成へ直接使用するのは、

``` text
text
meta.id
meta.title
meta.url
```

です。

残りのmetadataは、元データの由来やrevisionを記録するための情報です。

------------------------------------------------------------------------

# 10. Wikipedia固有metadata

## `meta.revision_id`

Wikipedia revision IDです。

``` json
"revision_id": "107436472"
```

どの版の記事からJSONLを生成したかを識別できます。

現在のRAG生成処理では直接使用しません。

------------------------------------------------------------------------

## `meta.timestamp`

revisionの更新日時です。

``` json
"timestamp": "2025-12-01T03:27:24Z"
```

現在のRAG生成処理では直接使用しません。

元記事確認用データベースなどでは利用できます。

------------------------------------------------------------------------

## `meta.sha1`

Wikipedia dumpに記録されているrevision本文のSHA-1です。

``` json
"sha1": "..."
```

元データの識別・検証用metadataで、現在のRAG生成処理では直接使用しません。

------------------------------------------------------------------------

## `meta.redirect`

リダイレクト記事かどうかを示します。

現在の `wikipediadump_xml-to-jsonl`
はリダイレクト記事を出力対象から除外しているため、出力される通常記事では、

``` json
"redirect": false
```

です。

------------------------------------------------------------------------

## `meta.redirect_target`

リダイレクト先です。

現在はリダイレクト記事自体を出力しないため、

``` json
"redirect_target": null
```

になります。

------------------------------------------------------------------------

## `meta.source`

データの種類を表します。

現在のWikipedia変換では、

``` json
"source": "wikimedia-pages-articles"
```

です。

RAG生成処理では直接使用しません。

------------------------------------------------------------------------

## `meta.source_dump`

どのWikipedia dumpから作成したかを記録します。

``` json
"source_dump": "jawiki-latest-pages-articles.xml.bz2"
```

RAG生成処理では直接使用しませんが、後からデータの由来を確認するときに有用です。

------------------------------------------------------------------------

# 11. 必須・推奨・任意の整理

現在の実装と実際の用途を分けて整理すると、次のようになります。

  項目                     builder上   実用上         用途
  ------------------------ ----------- -------------- --------------------
  `text`                   省略可      **必須**       RAG本文
  `meta`                   省略可      **必須相当**   metadata
  `meta.id`                省略可      **強く推奨**   文書識別
  `meta.title`             省略可      **必須相当**   タイトル検索・表示
  `meta.url`               省略可      任意           元文書URL
  `meta.revision_id`       未使用      任意           Wikipedia revision
  `meta.timestamp`         未使用      任意           更新日時
  `meta.sha1`              未使用      任意           元データ検証
  `meta.redirect`          未使用      任意           redirect情報
  `meta.redirect_target`   未使用      任意           redirect先
  `meta.source`            未使用      任意           データ種別
  `meta.source_dump`       未使用      任意           元dump識別

ここで「builder上で省略可」となっているのは、入力検証でエラーにしないという意味です。

良いRAGを作るための入力仕様としては、

``` text
text
id
title
```

を揃えることを推奨します。

------------------------------------------------------------------------

# 12. 独自JSONLを作る場合

`generate-RAG-from-jsonl`
という名前の通り、入力データはWikipediaに限定されません。

たとえば社内文書を、

``` jsonl
{"text":"システムAのバックアップ手順は...","meta":{"id":"manual-001","title":"システムA 運用マニュアル","url":""}}
{"text":"障害発生時には最初に...","meta":{"id":"manual-002","title":"障害対応手順","url":""}}
```

のようにすれば、同じ基本構造でRAG生成処理へ渡せます。

ただし、現在のchunkingや検索システムはWikipedia記事を使って調整しているため、別種類のデータでは、

-   chunk size
-   overlap
-   titleの付け方
-   Embeddingモデル
-   検索方式

を再評価した方がよい場合があります。

JSONL形式が互換であることと、現在のWikipedia向け設定がそのデータに最適であることは別の問題です。

------------------------------------------------------------------------

# 13. 1レコードの単位

基本的には、

``` text
1 JSON object = 1文書
```

と考えます。

Wikipediaでは、

``` text
1 JSON object = 1記事
```

です。

RAG生成側が、この1文書をさらに複数のchunkへ分割します。

したがって、入力JSONLをあらかじめRAG用の細かなchunkへ分割する必要はありません。

``` text
JSONL
  1記事
     ↓
ChunkBuilder
     ↓
chunk 0
chunk 1
chunk 2
...
```

という役割分担です。

独自データでも、意味的に一つの文書として扱いたい範囲を1レコードにするのが基本です。

------------------------------------------------------------------------

# 14. 本文の前処理

`generate-RAG-from-jsonl` は、入力された `text`
を「すでに本文として利用できる文章」として扱います。

そのため、

``` text
HTML → プレーンテキスト
Wikitext → プレーンテキスト
PDF → 本文抽出
不要なnavigationやfooterの除去
```

などは、原則としてJSONLを作る側の責務です。

Wikipediaの場合、この処理を `wikipediadump_xml-to-jsonl`
が担当しています。

``` text
Wikipedia XML / Wikitext
        ↓
wikipediadump_xml-to-jsonl
        ↓
cleaned text JSONL
        ↓
generate-RAG-from-jsonl
        ↓
chunking / embedding / index
```

`generate-RAG-from-jsonl` は汎用的なmarkup cleanerではありません。

------------------------------------------------------------------------

# 15. JSONLを作るときのチェックポイント

RAG生成前に、少なくとも次を確認すると安全です。

1.  1行が1つのJSON objectになっている
2.  UTF-8で保存されている
3.  各レコードに `text` がある
4.  `text` が空になっていない
5.  各レコードに `meta.title` がある
6.  `meta.id` が文書ごとに一意になっている
7.  同じ文書が大量に重複していない
8.  HTML / Wikitext等の不要なmarkupが残りすぎていない
9.  `.bz2` にする場合も、中身は同じJSONLである
10. 1文書を事前に細かなRAG chunkへ分割していない

------------------------------------------------------------------------

# 16. まとめ

最も単純には、`generate-RAG-from-jsonl` の入力は次の形式です。

``` json
{
  "text": "検索対象となる本文",
  "meta": {
    "id": "一意な文書ID",
    "title": "文書タイトル",
    "url": "元文書URL（なければ空文字列）"
  }
}
```

これを1行1文書で並べたJSONL、またはそのbzip2圧縮版を入力します。

``` text
document
   ↓
JSONL record
   ↓
ChunkBuilder
   ↓
chunks
   ↓
Embedding
   ↓
FAISS + SQLite
```

Wikipedia用の追加metadataは、元データのrevisionや出典を保持するためのもので、現在のRAG生成そのものに必須ではありません。

したがって、別データを `generate-RAG-from-jsonl` でRAG化するときも、まず

``` text
text
meta.id
meta.title
meta.url
```

という基本形へ変換することが、入力データ作成の基準になります。
