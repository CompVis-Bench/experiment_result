"""Offline evaluation of one model's predictions; no HTTP or credentials needed."""
import argparse
from collections import Counter
import json
from pathlib import Path

# Support both `python path/to/script.py` and `python -m package.module`.
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.evaluation_utils import ROOT, Contract, Dataset, implementation_hashes, input_path, read, sha, write
from eval.align_and_score import METRICS, VERSION, SearchLimitError, rates, score


def aggregate(rows):
    if len({r['score']['metric_version'] for r in rows}) > 1:
        raise ValueError('Cannot aggregate different metric versions')
    metrics = tuple(rows[0]['score']['metrics']) if rows else METRICS
    result = dict(sample_count=len(rows), statuses=dict(Counter(r['status'] for r in rows)),
                exact_match_count=sum(r['score']['exact_match'] for r in rows),
                exact_match_rate=sum(r['score']['exact_match'] for r in rows) / len(rows) if rows else None,
                metrics={name: rates(*(sum(r['score']['metrics'][name][k] for r in rows)
                                      for k in ('tp', 'predicted', 'reference'))) for name in metrics})
    if rows and 'unit_exact_match' in rows[0]['score']:
        correct = sum(r['score']['unit_exact_match']['correct'] for r in rows)
        aligned = sum(r['score']['unit_exact_match']['aligned'] for r in rows)
        result['unit_exact_match'] = dict(correct=correct, aligned=aligned,
                                         rate=correct / aligned if aligned else None)
    return result


def evaluate(dataset, contract, predictions, output, sample_ids=None, max_candidates=1000000,
             *, source_scores=None):
    if (predictions is None) == (source_scores is None):
        raise ValueError('Specify one of predictions or source_scores')
    predictions = Path(predictions) if predictions is not None else None
    output = Path(output)
    sources = None
    if source_scores is not None:
        source_scores = Path(source_scores).resolve()
        if source_scores == (output / 'scores.json').resolve():
            raise ValueError('Use a new output directory; do not overwrite source scores')
        source = read(source_scores)
        if source['dataset_sha256'] != dataset.sha256:
            raise ValueError('Source scores belong to a different dataset')
        sources = {r['sample_id']: r for r in source['samples']}
        if len(sources) != len(source['samples']) or not sources:
            raise ValueError('Source scores must have unique, nonempty samples')
    available = set(sources) if sources is not None else {r['sample_id'] for r in dataset.samples}
    selected = set(sample_ids) if sample_ids is not None else available
    if not selected or not selected <= available:
        raise ValueError('Selected sample absent from source scores or empty selection')
    if not selected <= {r['sample_id'] for r in dataset.samples}:
        raise ValueError('Selected sample not in dataset')
    rows, errors = [], []
    for row in dataset.samples:
        sid = row['sample_id']
        if sid not in selected:
            continue
        record = dict(sample_id=sid, target_sha256=row['target_sha256'])
        if sources is not None:
            saved = sources[sid]
            provenance = saved['provenance']
            path = ROOT / provenance['prediction_file']
            run_path = ROOT / provenance['run_file']
            # Provenance failures are fatal, not bad model predictions.
            if (saved['target_sha256'] != row['target_sha256']
                    or sha(path.read_bytes()) != saved['prediction_sha256']
                    or saved['prediction_sha256'] != provenance['prediction_sha256']
                    or sha(run_path.read_bytes()) != provenance['run_sha256']):
                raise RuntimeError(f'{sid}: source prediction provenance mismatch')
            record['provenance'] = provenance
        else:
            folder = predictions/'samples'/sid
            path = folder/'prediction.json'
            run_path = folder/'run.json'
            if not path.exists():
                path = predictions/(sid+'.json')
        reference = dataset.target(row)
        status = 'valid'
        try:
            if run_path.exists():
                run = read(run_path)
                for key in ('image_sha256', 'target_sha256'):
                    if run.get(key) != row[key]:
                        raise RuntimeError(f'{sid}: run provenance mismatch: {key}')
                if run.get('status') != 'valid':
                    status = run.get('status', 'invalid')
                    raise ValueError('Runner did not finish with a valid complete response')
            prediction = contract.validate(read(path))
            record['prediction_sha256'] = sha(path.read_bytes())
        except (OSError, ValueError) as exc:
            status = ('missing' if isinstance(exc, FileNotFoundError) else 'invalid') if status == 'valid' else status
            record['error'] = str(exc)
            prediction = {'charts': []}
        trace_path = output/'alignments'/(sid+'.jsonl')
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with trace_path.open('w', encoding='utf-8') as trace:
                result = score(prediction, reference, max_candidates,
                               trace=lambda item: trace.write(json.dumps(item, ensure_ascii=False)+'\n'))
        except SearchLimitError as exc:
            record.update(status='scoring_error', prediction_status=status, error=str(exc))
            errors.append({'sample_id': sid, 'error': str(exc)})
            rows.append(record)
            continue  # Retain other samples; never fabricate a zero or approximate score.
        if status != 'valid':
            result['exact_match'] = False
        record.update(status=status, score=result)
        rows.append(record)
    scored_rows = [row for row in rows if 'score' in row]
    report = dict(metric_version=VERSION, dataset_sha256=dataset.sha256,
                  scoring_complete=not errors, scoring_errors=errors,
                  selected_sample_count=len(rows), scored_sample_count=len(scored_rows),
                  implementation_sha256=implementation_hashes(),
                  denominator_policy=('All selected images for Full EM and mark recovery; encoding and Unit EM '
                      'use aligned units; sharing uses aligned endpoints only; sharing_all_groups retains all groups. '
                      'Internal and external fields are unioned for sharing. Undefined rates are null.'),
                  subsets={'scored_only' if errors else 'all': aggregate(scored_rows)}, samples=rows)
    if errors:
        write(output/'evaluation_error.json', dict(complete=False, errors=errors))
    else:
        (output/'evaluation_error.json').unlink(missing_ok=True)
    if sources is not None:
        report['source_scores'] = dict(path=str(source_scores), sha256=sha(source_scores.read_bytes()))
        if 'model' in source:
            report['model'] = source['model']
    write(output/'scores.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT/'data')
    p.add_argument('--schema', type=Path, help='Output schema; defaults to the dataset-declared schema')
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--predictions', type=Path, help='Directory containing saved model predictions')
    inputs.add_argument('--source-scores', type=Path, help='Merged scores.json with pinned per-sample prediction provenance')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sample-ids', nargs='+')
    p.add_argument('--max-candidates', type=int, default=1000000, help='0 = unlimited exact search; overflow is a fatal evaluation error')
    args = p.parse_args()
    if args.max_candidates < 0:
        p.error('max-candidates must be nonnegative')
    contract = Contract(input_path(args.dataset, 'schema', args.schema))
    report = evaluate(Dataset(args.dataset, contract), contract, args.predictions, args.output,
                      args.sample_ids, args.max_candidates, source_scores=args.source_scores)
    print(json.dumps(report['subsets'], ensure_ascii=False, indent=2))
    return 0 if report['scoring_complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
