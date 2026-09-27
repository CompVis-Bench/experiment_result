# Anonymous experiment results

This repository contains the compact, complete evaluation archive for the
latest 1,600-sample release. It contains 12 models × 1,600 images = 19,200
valid and scored model samples. The paired images and annotations are published
in the anonymous [dataset repository](https://github.com/CompVis-Bench/dataset);
this repository contains model outputs and scores only.

The repository is prepared for double-blind review. Its text, metadata, and
commit history contain no author names, contact details, institutions, local
paths, API keys, raw HTTP requests, or raw HTTP responses. Model names and
returned model identifiers are retained because they identify the evaluated
systems rather than the authors.

## Project links

- Project page: https://compvis-bench.github.io/
- Randomized renderer: https://compvis-bench.github.io/randomized_renderer/

## Contents

- `comparison.csv` / `comparison.json`: aggregate evaluator metrics.
- `models/<model>/predictions.jsonl.gz`: one parsed prediction per sample.
- `models/<model>/runs.jsonl.gz`: one run summary per sample, including usage,
  latency, status, and input/output hashes.
- `models/<model>/scores.json.gz`: the complete detailed score report.
- `inputs/`: the dataset manifest, schema, prompt, system instruction, model
  parameters, and scoring denominator policy.
- `provenance.json`: archive-level input and score hashes without local source
  paths.
- `archive_manifest.json`: SHA-256 checksums for every published file.
- `eval/align_and_score.py`: exact chart and field matching plus metric
  calculation.
- `eval/score_predictions.py`: scorer for unpacked prediction directories.
- `evaluate_archive.py`: scorer for the compact `predictions.jsonl.gz` files
  published here.
- `requirements.txt`: the offline scoring dependencies.

## Recompute a model score

Clone the anonymous dataset repository next to this repository, install the
two packages in `requirements.txt`, and run a smoke test on one or more sample
IDs:

```bash
python3 -m pip install -r requirements.txt
python3 evaluate_archive.py \
  --dataset ../dataset/new-1600 \
  --archive . \
  --model gpt-6-astra \
  --sample-ids composite-01-001 \
  --output /tmp/compvis-score-smoke
```

Omit `--sample-ids` to score all 1,600 samples. The evaluator first validates
the dataset manifest and output schema, then performs exact maximum-cardinality
chart matching, exact field assignment, sharing matching, and metric
aggregation. `alignments/` contains the exhaustive optimal-structure trace for
the selected samples; it is a generated output and is not part of this archive.

## Integrity check

From this repository root, verify the archive files with:

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('.')
manifest = json.loads((root / 'archive_manifest.json').read_text())
for item in manifest['record_files']:
    p = root / item['path']
    assert p.is_file(), item['path']
    assert p.stat().st_size == item['bytes'], item['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], item['path']
print('archive files verified:', len(manifest['record_files']))
PY
```

For compressed records, use `gzip -dc`. Each model directory has 1,600
predictions, 1,600 run records, and 1,600 detailed score records.

This is an anonymous supplementary-results repository for the review period.
Add formal attribution only after the review period, when the venue permits it.
