# generate-RAG-from-jsonl

JSONL形式の記事データから、ローカルLLM +
RAGシステムで使用する検索データを生成するプロジェクトです。

現在の標準構成では、Wikipediaから生成した `ja_wiki.jsonl.bz2`
を入力し、ruri-v3 (`ruri-embed`)
でEmbeddingを作成し、FAISSのshardとSQLiteメタデータを `../data/index/`
に生成します。

このREADMEは、**現在実際に使用しているRAGと同じ設定で再生成する場合**の手順を中心に記載します。設定の選定・再評価・カスタマイズについては
`RESEARCH_GUIDE.md` を参照してください。

## ディレクトリ構成

想定する配置は次のとおりです。

``` text
RAG/
├─ generate-RAG-from-jsonl/
│  ├─ build_production.py
│  ├─ build.cmd
│  ├─ chunker.py
│  ├─ embedding.py
│  ├─ faiss_helpers.py
│  ├─ models.py
│  ├─ sqlite_store.py
│  ├─ utils.py
│  │
│  ├─ configs/
│  │  ├─ build_config.json
│  │  ├─ faiss_config.json
│  │  └─ faiss_candidates.json
│  │
│  ├─ sample/
│  │  ├─ sample.sqlite
│  │  ├─ sample_vectors.f32
│  │  ├─ sample_ids.i64
│  │  └─ sample_vectors.json
│  │
│  └─ research/
│     ├─ sampling/
│     ├─ benchmarking/
│     └─ diagnostics/
│
└─ data/
   ├─ jsonl/
   │  └─ ja_wiki.jsonl.bz2
   └─ index/
```

`data` はリポジトリの外に置きます。Wikipedia
JSONLや生成済みindexはサイズが大きいため、Git管理対象にする必要はありません。

## 現行RAGと同じものを生成するために必要なもの

最低限、次を確認してください。

1.  `../data/jsonl/ja_wiki.jsonl.bz2` が存在する
2.  Ollamaがインストールされ、起動できる
3.  Ollamaに `ruri-embed` が存在する
4.  Pythonから `ollama`, `faiss-cpu`, `numpy`, `tqdm` を利用できる
5.  `configs/build_config.json` と `configs/faiss_config.json`
    を変更しない
6.  `sample/sample_vectors.f32` と `sample/sample_vectors.json`
    が存在する
7.  新規に最初から生成する場合、既存の `../data/index/`
    を上書きしてよいことを確認する

現在のEmbedding設定は次のとおりです。

``` json
{
  "embedding_model": "ruri-embed",
  "embedding_dimension": 768,
  "query_prefix": "検索クエリ: ",
  "document_prefix": "検索文書: ",
  "target_chunk_chars": 1200,
  "max_chunk_chars": 1600,
  "overlap_chars": 200,
  "min_chunk_chars": 80,
  "embedding_batch_size": 16,
  "shard_articles": 50000,
  "save_float16": true
}
```

FAISSの現行設定は `configs/faiss_config.json`
に保存されています。現在採用している構成は `ivf8192_pq64` (`IVF-PQ`,
`nlist=8192`, `pq_m=64`, `pq_bits=8`) です。

## Pythonパッケージ

必要なPythonパッケージは少なくとも次の4つです。

``` cmd
py -m pip install ollama faiss-cpu numpy tqdm
```

標準ライブラリの `sqlite3`, `bz2`, `json` なども使用します。

## ruri-embed

EmbeddingにはOllama上の `ruri-embed` を使用します。

確認:

``` cmd
ollama list
```

現在のRAGと互換性を保つには、**RAG生成時と検索時で同じEmbeddingモデル、次元数、query/document
prefixを使用する必要があります。**

`ruri-embed`
はruri-v3-310mをOllamaから利用するために作成したモデル名です。既存環境に現在使用中の
`ruri-embed` がある場合は、それをそのまま使用してください。

## 入力JSONL

デフォルト入力:

``` text
../data/jsonl/ja_wiki.jsonl.bz2
```

1行1記事で、少なくとも次の形を想定します。

``` json
{"text":"記事本文","meta":{"id":"...","title":"...","url":"..."}}
```

`.jsonl` と `.jsonl.bz2` の両方を読み込めます。

## 生成前のsampleについて

production用FAISS indexを最初にtrainingするため、

``` text
sample/sample_vectors.f32
sample/sample_vectors.json
```

を使用します。

現在のsampleが残っており、現在と同じ設定でRAGを再生成するだけなら、sampleを作り直す必要はありません。

sampleがない場合は次の順に生成します。

``` cmd
research\sampling\01-sample.cmd
research\sampling\02-embed-sample.cmd
```

既存ファイルを作り直す場合:

``` cmd
research\sampling\01-sample.cmd --overwrite
research\sampling\02-embed-sample.cmd --overwrite
```

デフォルトでは全JSONLを走査し、reservoir samplingで最大500,000
chunkを抽出します。その後、そのsampleを `ruri-embed` でEmbeddingします。

## production build

現行設定のまま最初から生成する場合:

``` cmd
build.cmd --overwrite
```

または、

``` cmd
py build_production.py --overwrite
```

デフォルトでは、

``` text
入力: ../data/jsonl/ja_wiki.jsonl.bz2
出力: ../data/index/
```

となります。

### 生成される主なファイル

`../data/index/` には、おおむね次のものが生成されます。

``` text
metadata.sqlite
trained_template.faiss

shard_000000.faiss
shard_000000.f16
shard_000000.ids
shard_000000.json

shard_000001.faiss
...

config.json
progress.json
build.log
```

FAISS shardは検索用indexです。`metadata.sqlite`
にはchunk本文・タイトル・記事情報などを格納します。

`save_float16=true` のため、再ランキング等で元Embeddingを利用できるよう
`.f16` と対応する `.ids` も保存します。

## 中断と再開

production buildは完了済みshardを `metadata.sqlite` の
`completed_shards` で管理し、入力JSONL上のoffsetを記録します。

通常の中断後は、同じコマンドをもう一度実行することで完了済みshardの続きから再開できます。

``` cmd
build.cmd
```

**再開時には `--overwrite` を付けないでください。**

`--overwrite` は既存indexを削除して最初から作り直す場合に使用します。

進捗は、

``` text
../data/index/progress.json
../data/index/build.log
```

でも確認できます。

## SQLite indexだけをfinalizeする

データ生成自体が完了していて、SQLiteの検索用index作成だけを実行したい場合:

``` cmd
build.cmd --finalize-only
```

通常のproduction
buildでは最後に自動的にfinalizeされるため、通常は必要ありません。

## 入出力場所を変更する

必要ならコマンドラインで変更できます。

``` cmd
py build_production.py ^
  --input D:\data\my_articles.jsonl.bz2 ^
  --index-dir D:\data\my_index
```

設定ファイルも明示できます。

``` cmd
py build_production.py ^
  --build-config configs\build_config.json ^
  --index-config configs\faiss_config.json
```

## 現行設定をそのまま再現する場合に変更しない方がよいもの

特に次は、検索側との互換性や検索結果そのものに影響します。

-   `embedding_model`
-   `embedding_dimension`
-   `query_prefix`
-   `document_prefix`
-   chunkサイズ関連設定
-   `index_type`
-   `nlist`
-   `pq_m`
-   `pq_bits`

これらを変更する場合は「同じRAGの再生成」ではなく、新しいRAG構成の評価と考え、`RESEARCH_GUIDE.md`
の手順でsample/benchmarkから確認することを推奨します。

## 関連プロジェクト

``` text
wikipediadump_xml-to-jsonl
    Wikipedia XML dump
        ↓
    JSONL

generate-RAG-from-jsonl
    JSONL
        ↓
    FAISS + SQLite + Embeddingデータ

LocalLLM_with_Wikipedia
    生成済みRAG
        ↓
    Wikipedia検索 + ローカルLLM
```

このリポジトリは中央の「JSONLからRAG検索データを生成する工程」だけを担当します。
