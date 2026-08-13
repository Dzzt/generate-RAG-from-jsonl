# generate-RAG-from-jsonl Research / Tuning Guide

この文書は、`generate-RAG-from-jsonl`
の設定を変更して、自分のJSONLやPC環境に合わせたRAGを設計・評価するためのメモです。

単に現在使用中のWikipedia RAGを同じ設定で再生成する場合は `README.md`
だけで十分です。

------------------------------------------------------------------------

## 1. 調整工程の全体像

基本的には次の順序で評価します。

``` text
JSONL
  │
  ├─ build_config.json
  │      chunking / embedding条件
  │
  ▼
01_sample_chunks.py
  │
  ▼
sample.sqlite
  │
  ▼
02_embed_sample.py
  │
  ▼
sample_vectors.f32
  │
  ├─ faiss_candidates.json
  │
  ▼
03_benchmark_indexes.py
  │
  ▼
benchmark_results.json
  │
  ▼
faiss_config.json
  │
  ▼
build_production.py
  │
  ▼
production RAG
```

大きく、

1.  **文章をどうchunkにするか**
2.  **何でEmbeddingするか**
3.  **EmbeddingをFAISSでどう圧縮・検索するか**
4.  **検索時にどこまで候補を拾うか**

の4段階があります。

------------------------------------------------------------------------

# 2. configs/build_config.json

現在値:

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

## embedding_model

Ollamaで使用するEmbeddingモデル名です。

現在:

``` text
ruri-embed
```

Embeddingモデルを変更するとベクトル空間そのものが変わるため、**既存FAISS
indexとの互換性はありません。**

モデルを変更した場合は、

``` text
sample生成
→ sample embedding
→ FAISS benchmark
→ production build
```

を一通りやり直すのが安全です。

### 判断の目安

モデル変更を検討するのは例えば、

-   日本語固有名詞の検索が弱い
-   自然文queryと文書の対応が悪い
-   現在より小さいモデルで高速化したい
-   より大きなEmbeddingモデルで検索品質を上げたい

といった場合です。

単にLLMの回答品質が悪い場合は、Embeddingモデルより先に「検索結果に正しいchunkが入っているか」を確認してください。正しいchunkが取得できているなら、問題はRAG生成側ではなく検索後の選別や回答生成側にある可能性があります。

## embedding_dimension

Embeddingの次元数です。

現在:

``` text
768
```

これは使用するEmbeddingモデルの出力次元と一致している必要があります。

モデルを変更せずにこの値だけを変更してはいけません。

## query_prefix / document_prefix

現在:

``` text
query_prefix    = "検索クエリ: "
document_prefix = "検索文書: "
```

ruri系Embeddingでqueryとdocumentの役割を区別するために使用しています。

重要なのは、production RAG生成時だけでなく、**検索時にも同じquery
prefixを使用すること**です。

prefixを変更した場合は、原則として文書Embeddingも作り直して比較してください。

------------------------------------------------------------------------

# 3. chunk設定

## target_chunk_chars

現在:

``` text
1200
```

chunkの基本的な目標サイズです。

### 小さくすると

-   一つのchunkが扱う話題が限定されやすい
-   ピンポイント検索では有利になる場合がある
-   chunk数が増える
-   Embedding生成時間が増える
-   FAISS/SQLite/float16データが大きくなる
-   文脈が複数chunkに分断されやすくなる

### 大きくすると

-   一つのchunkに広い文脈を保持できる
-   chunk数が減る
-   buildが軽くなる
-   1 chunkに複数話題が混ざり、Embeddingの焦点がぼやける場合がある
-   Embeddingモデルのcontext上限に近づきやすくなる

### 判断例

検索すると「目的の記事は出るが、欲しい記述のあるchunkが上位に来ない」ことが多いなら、chunkを少し小さくする実験に意味があります。

逆に「細切れの断片ばかり取得され、前後関係が不足する」なら大きくする方向が候補です。

一度に大きく変更するより、

``` text
1200 → 1000
1200 → 1400
```

程度から比較する方が原因を追いやすくなります。

## max_chunk_chars

現在:

``` text
1600
```

chunkが肥大化した場合の上限です。

`target_chunk_chars`
を変更するときは、こちらとの関係も確認してください。

targetとmaxを極端に近づけると分割の自由度が小さくなり、逆にmaxだけ非常に大きくすると一部のchunkだけ長くなる可能性があります。

## overlap_chars

現在:

``` text
200
```

隣接chunk間に重複させる文字数です。

### 増やすと

-   chunk境界付近の情報を拾いやすくなる
-   同じ文章が複数chunkに入る
-   検索結果が似たchunkで埋まりやすくなる
-   chunk総量・Embedding量が増える

### 減らすと

-   データ量は減る
-   重複検索結果も減る
-   境界をまたぐ説明が検索しにくくなる場合がある

検索結果に「ほぼ同じ隣接chunk」が大量に並ぶなら、overlapを減らす実験が候補です。

逆に、節の境界付近にある情報だけ取り逃しやすいなら増やす余地があります。

## min_chunk_chars

現在:

``` text
80
```

極端に短い断片を独立chunkとして残さないための値です。

検索結果に見出しだけ、短い断片だけ、といった情報量の少ないchunkが多い場合は上げる方向を検討できます。

ただし、Wikipediaの短い節や短い記事そのものを落としすぎないよう注意します。

------------------------------------------------------------------------

# 4. embedding_batch_size

現在:

``` text
16
```

一度にOllamaへ送るEmbedding件数です。

### 大きくする

-   GPU/CPU/RAMに余裕があればthroughputが上がる可能性がある
-   context-lengthエラーやメモリ不足が起きやすくなる場合がある

### 小さくする

-   安定しやすい
-   呼び出し回数が増え、遅くなる可能性がある

現在の `embedding.py`
はcontext-length系エラーの場合、batchを分割し、1件でも長すぎる場合は入力を段階的に短縮する防御処理を持っています。

`build.log` に長文短縮やEmbedding失敗が頻繁に出る場合は、

1.  chunkサイズ
2.  Embeddingモデルのcontext
3.  batch size

の順で状況を確認します。

batch sizeを下げても「1 chunk自体が長すぎる」問題は解決しません。

------------------------------------------------------------------------

# 5. shard_articles

現在:

``` text
50000
```

production buildで1 shardに処理する記事数です。

### 小さくする

-   1 shard完成までが短くなる
-   中断時の心理的・運用上の扱いが楽になる
-   shardファイル数が増える
-   検索側で扱うindex数も増える

### 大きくする

-   shard数が減る
-   1 shardあたりの処理時間・一時的なメモリ使用量が増える

検索品質そのものを改善する設定ではありません。主にbuild運用・メモリ・ファイル数の調整です。

現在のPCで50,000記事/shardが安定しているなら、検索品質のために変更する必要はありません。

------------------------------------------------------------------------

# 6. save_float16

現在:

``` text
true
```

FAISSへ格納する圧縮indexとは別に、元Embeddingをfloat16で保存します。

これにより、検索側で候補を取り出した後に元ベクトルに近いデータを使った再ランキングが可能になります。

`false` にするとディスク使用量を削減できますが、float16
rerankを利用できなくなります。

現在の `faiss_config.json` は、

``` json
"use_float16_rerank": true
```

なので、現在と同じ検索方式を維持するなら `save_float16=true`
のままにします。

------------------------------------------------------------------------

# 7. sampleを作り直す

chunk/Embedding条件を変えた場合はsampleを作り直します。

``` cmd
research\sampling\01-sample.cmd --overwrite
research\sampling\02-embed-sample.cmd --overwrite
```

## 01_sample_chunks.py

全JSONLを走査し、reservoir samplingでchunkを抽出します。

デフォルト:

``` text
target-chunks = 500000
seed          = 20260721
```

500,000という値は、現在比較しているFAISS候補のtrainingにも関係します。

FAISS benchmarkは `39 × nlist` をtraining
vector数の一つの目安として警告表示します。

現在候補では、

``` text
nlist=8192   → 319,488
nlist=12288  → 479,232
```

なので、500,000 sampleなら両方をほぼ十分な規模でtrainingできます。

nlistをさらに増やす場合は、sample数も増やす必要がないか確認してください。

------------------------------------------------------------------------

# 8. FAISS候補: configs/faiss_candidates.json

現在は次の3候補を比較します。

``` text
ivf8192_pq64
ivf12288_pq64
ivf12288_sq8
```

## nlist

IVFでベクトル空間をいくつのclusterに分けるかです。

### 増やすと

-   各clusterが細かくなる
-   適切なnprobeなら検索効率・精度が改善する可能性がある
-   trainingにより多くのsampleが必要
-   nprobeが小さすぎると、細分化したのに必要clusterを探索せずrecallが落ちる場合がある

### 減らすと

-   trainingが軽くなる
-   少ないsampleでも扱いやすい
-   clusterが粗くなる

`nlist` 単独では判断せず、`nprobe` とセットでbenchmarkします。

## PQ と SQ8

### IVFPQ

現在採用中です。

PQ (Product Quantization)
でベクトルを強く圧縮するため、indexを小さくできます。

代償として近似誤差があります。

`pq_m=64`, `pq_bits=8` は、768次元ベクトルを64
sub-vectorに分割して量子化する現在の設定です。

### IVFSQ8

Scalar Quantizationで各成分を8bit化します。

一般にPQよりindexサイズは大きくなりやすい一方、量子化の性質が異なるためrecallとのトレードオフを比較する候補になります。

「ディスク容量よりrecallを優先したい」場合には、SQ8候補をbenchmarkに残して比較する意味があります。

------------------------------------------------------------------------

# 9. benchmarkを実行する

sample embeddingを作成した後、

``` cmd
research\benchmarking\03-benchmark.cmd
```

を実行します。

デフォルトでは、

``` text
training vectors = 最大500,000
evaluation       = 最大100,000
queries          = 200
```

を使用します。

各FAISS候補について、

``` text
nprobe = 32 / 64 / 128 / 256
candidate_count = 100 / 300 / 500
```

を比較します。

結果:

``` text
research/benchmarking/benchmarks/benchmark_results.json
```

選択された設定:

``` text
configs/faiss_config.json
```

------------------------------------------------------------------------

# 10. benchmark結果の読み方

主に見る値は、

``` text
recall_at_10_in_candidates
query_ms_mean
index_size_bytes
```

です。

## recall_at_10_in_candidates

exact search (`IndexFlatIP`) のtop
10が、近似FAISS検索で取り出した候補集合にどれだけ含まれているかです。

1.0に近いほど、近似indexによる取りこぼしが少ないことを意味します。

現在の自動選定では、

``` text
recall >= 0.95
```

を合格ラインとしています。

合格候補の中から、

1.  indexサイズが小さい
2.  queryが速い
3.  recallが高い

の順で選びます。

現在の選択結果:

``` text
index_type      = ivfpq
nlist           = 8192
pq_m            = 64
pq_bits         = 8
nprobe          = 256
candidate_count = 300
sample_recall   ≈ 0.957
sample_query_ms ≈ 0.182 ms
```

### 重要

このbenchmarkは**RAG質問に対する最終回答品質を直接測るものではありません。**

sample内のEmbeddingをqueryとしてexact
FAISSと近似FAISSの差を測っています。

したがって、

``` text
recall 0.96 → LLM回答が96%正しい
```

という意味ではありません。

これは「FAISS近似検索が、元Embedding空間で見つかる近傍をどの程度取りこぼしているか」の診断です。

------------------------------------------------------------------------

# 11. 結果別の調整例

## recallが0.95未満

候補:

1.  `nprobe` を増やす
2.  `candidate_count` を増やす
3.  `nlist` の異なる候補を試す
4.  PQではなくSQ8を試す
5.  PQの圧縮条件を変更する

まずはindex構造を作り直さずに比較できる `nprobe` / `candidate_count`
の範囲を広げるのが簡単です。

例:

``` text
nprobe 256でrecall 0.94
nprobe 512でrecall 0.97
```

となるなら、検索速度との交換で512を採用する選択肢があります。

ただし現在のbenchmarkコードは256までなので、512を試す場合はbenchmarkのテスト値を追加します。

## recallは十分だが検索が遅い

候補:

-   `nprobe` を下げる
-   `candidate_count` を下げる
-   より小さいnlist構成も比較する

例:

``` text
nprobe=256  recall=.957  0.18 ms
nprobe=128  recall=.953  0.11 ms
```

であれば、0.95を維持できる128の方が実用上有利な場合があります。

## recallは高いがindexが大きい

PQ候補を優先する、`pq_m`
など圧縮条件を再評価する、といった方向になります。

ただし圧縮を強くするとrecallが落ちるため、必ずbenchmarkで確認します。

## benchmarkは良いのに実際の質問で関係ない記事が出る

これはFAISS近似誤差ではなく、

-   Embeddingモデル
-   chunkの作り方
-   queryの書き方
-   タイトル一致/キーワード検索との融合
-   rerank
-   検索結果をLLMへ渡す前の選別

を疑うべきです。

exact近傍を0.95以上拾えているなら、「FAISSが目的ベクトルを落としている」以外の問題である可能性が高くなります。

特にWikipediaでは、人物名・作品名などのタイトル検索と自然文Embedding検索をどう組み合わせるかが、最終的なRAG品質に大きく影響します。

------------------------------------------------------------------------

# 12. nprobe と candidate_count は性質が違う

## nprobe

IVFのclusterを何個探索するかです。

増やすほど探索範囲が広がり、通常recallは上がりやすくなりますが検索時間も増えます。

## candidate_count

FAISSから後段へ渡す候補件数です。

float16
rerankなど後段処理を行う場合、候補を増やすほど本当に近いvectorを救える可能性がありますが、後段処理量が増えます。

この2値はEmbeddingを作り直さずに調整できる検索側パラメータです。

ただしproduction generatorは採用値を `data/index/config.json`
に記録するため、実際の検索プログラムがどの設定を読むかも合わせて確認してください。

------------------------------------------------------------------------

# 13. FAISSの構造を変えた場合

次を変更した場合:

``` text
index_type
nlist
pq_m
pq_bits
```

FAISS indexそのものの構造が変わるため、production indexは再生成します。

一方、

``` text
nprobe
candidate_count
```

は検索時の探索量なので、FAISS
vector自体を再Embeddingする必要はありません。

------------------------------------------------------------------------

# 14. chunk設定を変更した場合

次を変更した場合:

``` text
target_chunk_chars
max_chunk_chars
overlap_chars
min_chunk_chars
```

chunkの個数・本文・chunk IDが変わります。

そのため、

``` text
sample.sqlite
sample_vectors.*
production FAISS
metadata.sqlite
```

を一式作り直すのが安全です。

------------------------------------------------------------------------

# 15. Embeddingモデルを変更した場合

次を変更した場合:

``` text
embedding_model
embedding_dimension
query_prefix
document_prefix
```

sample vectorsからproduction indexまで作り直します。

検索側も同じモデル・prefixへ変更します。

異なるEmbeddingモデルで生成したvectorを同じFAISS
indexに混在させてはいけません。

------------------------------------------------------------------------

# 16. production build前の推奨確認

新しい設定を試す場合は、いきなりWikipedia全件を再Embeddingするより、

1.  `build_config.json` を変更
2.  sampleを再生成
3.  sampleをEmbedding
4.  benchmark
5.  `benchmark_results.json` を確認
6.  `faiss_config.json` を確認
7.  必要なら実際の検索用コードで小規模な検索品質を確認
8.  問題なければproduction build

という順序が安全です。

FAISS
benchmarkだけでは「意味検索として良いか」は判定できないので、最終的には実際に検索したい質問群で確認します。

------------------------------------------------------------------------

# 17. 現在設定を基準点として残す

チューニング時には、一度に複数項目を変えない方が比較しやすくなります。

例えばchunkを試すなら、

``` text
A: target=1000 overlap=200
B: target=1200 overlap=200  ← 現在
C: target=1400 overlap=200
```

のように、まず一つだけ変えます。

FAISSなら、

``` text
同じsample vectors
    ↓
複数のFAISS候補
```

とすることで、Embedding品質とFAISS近似品質を混同せずに比較できます。

現在の設定は実運用できている基準点なので、`build_config.json` と
`faiss_config.json` はGitで履歴を残してから変更するのが適しています。
