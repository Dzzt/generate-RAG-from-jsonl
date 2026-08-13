#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faiss_helpers import create_index, train_index
from utils import read_json, write_json

DEFAULT_VECTORS = PROJECT_ROOT / "sample" / "sample_vectors.f32"
DEFAULT_MANIFEST = PROJECT_ROOT / "sample" / "sample_vectors.json"
DEFAULT_CANDIDATES = PROJECT_ROOT / "configs" / "faiss_candidates.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "benchmarks"
DEFAULT_SELECTED_CONFIG = PROJECT_ROOT / "configs" / "faiss_config.json"


def recall_at(reference: np.ndarray, candidate: np.ndarray) -> float:
    total = 0.0
    for ref, cand in zip(reference, candidate):
        total += len(set(map(int, ref)) & set(map(int, cand))) / len(ref)
    return total / len(reference)


def enable_faiss_verbose(index) -> None:
    for obj, attr in ((index, "verbose"), (getattr(index, "cp", None), "verbose"),
                      (getattr(getattr(index, "pq", None), "cp", None), "verbose")):
        if obj is not None:
            try:
                setattr(obj, attr, True)
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--query-count", type=int, default=200)
    parser.add_argument("--train-count", type=int, default=500_000)
    parser.add_argument("--eval-count", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    args.vectors = args.vectors.resolve()
    args.manifest = args.manifest.resolve()
    args.candidates = args.candidates.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.vectors.is_file():
        parser.error(f"Sample vectors not found: {args.vectors}")
    if not args.manifest.is_file():
        parser.error(f"Sample manifest not found: {args.manifest}")
    if not args.candidates.is_file():
        parser.error(f"Candidates config not found: {args.candidates}")

    if args.threads > 0:
        faiss.omp_set_num_threads(args.threads)

    manifest = read_json(args.manifest)
    count = int(manifest["count"])
    dimension = int(manifest["dimension"])
    vectors = np.memmap(args.vectors, mode="r", dtype=np.float32, shape=(count, dimension))

    rng = np.random.default_rng(args.seed)

    train_count = min(args.train_count, count)
    train_rows = rng.choice(count, size=train_count, replace=False)
    print(f"Loading training vectors: {train_count:,} x {dimension}", flush=True)
    train_vectors = np.ascontiguousarray(vectors[train_rows], dtype=np.float32)

    eval_count = min(args.eval_count, count)
    eval_rows = rng.choice(count, size=eval_count, replace=False)
    print(f"Loading evaluation vectors: {eval_count:,} x {dimension}", flush=True)
    eval_vectors = np.ascontiguousarray(vectors[eval_rows], dtype=np.float32)

    query_count = min(args.query_count, eval_count)
    query_rows = rng.choice(eval_count, size=query_count, replace=False)
    queries = np.ascontiguousarray(eval_vectors[query_rows], dtype=np.float32)

    print("Building exact-search reference...", flush=True)
    flat = faiss.IndexFlatIP(dimension)
    flat.add(eval_vectors)
    _, exact_top10 = flat.search(queries, 10)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for candidate in read_json(args.candidates):
        name = candidate["name"]
        nlist = int(candidate["nlist"])
        recommended_minimum = 39 * nlist

        print(f"\n=== {name} ===", flush=True)
        print(
            f"Training vectors: {train_count:,}; "
            f"recommended minimum for nlist={nlist:,}: {recommended_minimum:,}",
            flush=True,
        )
        if train_count < recommended_minimum:
            print("WARNING: below FAISS recommended minimum.", flush=True)

        index = create_index(
            dimension=dimension,
            index_type=candidate["index_type"],
            nlist=nlist,
            pq_m=int(candidate.get("pq_m", 0)),
            pq_bits=int(candidate.get("pq_bits", 8)),
        )
        enable_faiss_verbose(index)

        print("Training index; this may take tens of minutes...", flush=True)
        started = time.perf_counter()
        train_index(index, train_vectors)
        train_seconds = time.perf_counter() - started
        print(f"Training completed: {train_seconds / 60:.1f} minutes", flush=True)

        print(f"Adding {eval_count:,} evaluation vectors...", flush=True)
        started = time.perf_counter()
        index.add(eval_vectors)
        add_seconds = time.perf_counter() - started
        print(f"Add completed: {add_seconds / 60:.1f} minutes", flush=True)

        index_path = args.output_dir / f"{name}.faiss"
        faiss.write_index(index, str(index_path))
        result = {
            **candidate,
            "train_count": train_count,
            "eval_count": eval_count,
            "train_seconds": train_seconds,
            "add_seconds": add_seconds,
            "index_size_bytes": index_path.stat().st_size,
            "tests": [],
        }

        for nprobe in (32, 64, 128, 256):
            index.nprobe = min(nprobe, nlist)
            for candidate_count in (100, 300, 500):
                started = time.perf_counter()
                _, found = index.search(queries, candidate_count)
                elapsed = time.perf_counter() - started
                test = {
                    "nprobe": nprobe,
                    "candidate_count": candidate_count,
                    "recall_at_10_in_candidates": recall_at(exact_top10, found),
                    "query_ms_mean": elapsed * 1000 / query_count,
                }
                result["tests"].append(test)
                print(test, flush=True)

        results.append(result)
        del index

    write_json(args.output_dir / "benchmark_results.json", results)

    passing = []
    fallback = []
    for result in results:
        for test in result["tests"]:
            pair = (result, test)
            fallback.append(pair)
            if test["recall_at_10_in_candidates"] >= 0.95:
                passing.append(pair)

    if passing:
        best_result, best_test = min(
            passing,
            key=lambda item: (
                item[0]["index_size_bytes"],
                item[1]["query_ms_mean"],
                -item[1]["recall_at_10_in_candidates"],
            ),
        )
    else:
        best_result, best_test = max(
            fallback,
            key=lambda item: (
                item[1]["recall_at_10_in_candidates"],
                -item[1]["query_ms_mean"],
                -item[0]["index_size_bytes"],
            ),
        )

    selected = {
        "name": best_result["name"],
        "index_type": best_result["index_type"],
        "nlist": best_result["nlist"],
        "pq_m": best_result.get("pq_m", 0),
        "pq_bits": best_result.get("pq_bits", 8),
        "nprobe": best_test["nprobe"],
        "candidate_count": best_test["candidate_count"],
        "use_float16_rerank": True,
        "sample_recall": best_test["recall_at_10_in_candidates"],
        "sample_query_ms": best_test["query_ms_mean"],
    }
    write_json(DEFAULT_SELECTED_CONFIG, selected)

    print("\nSelected:", flush=True)
    print(json.dumps(selected, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
