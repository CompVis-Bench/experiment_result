"""Score one compact model archive against the published dataset."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from eval.align_and_score import METRICS, SearchLimitError, VERSION, rates, score
from eval.evaluation_utils import Contract, Dataset, input_path, write


def read_jsonl_gz(path):
    records = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or sample_id in records:
                raise ValueError(f"duplicate or missing sample_id in {path}: {sample_id!r}")
            records[sample_id] = record
    return records


def aggregate(rows):
    metrics = tuple(rows[0]["score"]["metrics"]) if rows else METRICS
    return {
        "sample_count": len(rows),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "exact_match_count": sum(r["score"]["exact_match"] for r in rows),
        "exact_match_rate": (sum(r["score"]["exact_match"] for r in rows) / len(rows)
                              if rows else None),
        "metrics": {
            name: rates(*(sum(r["score"]["metrics"][name][key] for r in rows)
                          for key in ("tp", "predicted", "reference")))
            for name in metrics
        },
    }


def evaluate_archive(dataset_path, archive_path, model, output_path, sample_ids=None,
                      max_candidates=1_000_000):
    dataset_path = Path(dataset_path)
    archive_path = Path(archive_path)
    model_dir = archive_path / "models" / model
    predictions = read_jsonl_gz(model_dir / "predictions.jsonl.gz")
    runs = read_jsonl_gz(model_dir / "runs.jsonl.gz")
    contract = Contract(input_path(dataset_path, "schema"))
    dataset = Dataset(dataset_path, contract)
    selected = set(sample_ids) if sample_ids else {row["sample_id"] for row in dataset.samples}
    expected = {row["sample_id"] for row in dataset.samples}
    if not selected or not selected <= expected:
        raise ValueError("sample_ids must be nonempty and present in the dataset")
    if set(predictions) != expected:
        raise ValueError("prediction sample IDs do not exactly match the dataset")
    if set(runs) != expected:
        raise ValueError("run sample IDs do not exactly match the dataset")

    rows = []
    for row in dataset.samples:
        sample_id = row["sample_id"]
        if sample_id not in selected:
            continue
        prediction_record = predictions[sample_id]
        run_record = runs[sample_id]
        if run_record.get("status") != "valid":
            raise ValueError(f"{sample_id}: compact archive contains a non-valid run")
        if run_record.get("image_sha256") != row["image_sha256"]:
            raise ValueError(f"{sample_id}: image checksum mismatch")
        if run_record.get("target_sha256") != row["annotation_sha256"]:
            raise ValueError(f"{sample_id}: target checksum mismatch")
        payload = contract.validate(prediction_record["prediction"])
        trace_path = Path(output_path) / "alignments" / f"{sample_id}.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8") as trace:
            result = score(payload, dataset.target(row), max_candidates,
                           trace=lambda item: trace.write(json.dumps(item, ensure_ascii=False) + "\n"))
        rows.append({"sample_id": sample_id, "target_sha256": row["annotation_sha256"],
                     "prediction_sha256": prediction_record.get("prediction_sha256"),
                     "status": "valid", "score": result})
    report = {
        "metric_version": VERSION,
        "dataset_sha256": dataset.sha256,
        "model": model,
        "selected_sample_count": len(rows),
        "scored_sample_count": len(rows),
        "scoring_complete": True,
        "denominator_policy": ("All selected images for Full EM and mark recovery; encoding and Unit EM "
                               "use aligned units; sharing uses aligned endpoints only; sharing_all_groups "
                               "retains all groups. Internal and external fields are unioned for sharing. "
                               "Undefined rates are null."),
        "subsets": {"all": aggregate(rows)},
        "samples": rows,
    }
    write(Path(output_path) / "scores.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="dataset repository release, e.g. ../dataset/new-1600")
    parser.add_argument("--archive", type=Path, default=Path("."),
                        help="root of this experiment_result repository")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-ids", nargs="+", help="optional subset for a smoke test")
    parser.add_argument("--max-candidates", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.max_candidates < 0:
        parser.error("--max-candidates must be nonnegative")
    report = evaluate_archive(args.dataset, args.archive, args.model, args.output,
                              args.sample_ids, args.max_candidates)
    print(json.dumps(report["subsets"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
