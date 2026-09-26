"""Validate the final dataset or an unmodified model-output JSON file."""
import argparse
from pathlib import Path

# Support both `python path/to/script.py` and `python -m package.module`.
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.evaluation_utils import ROOT, Contract, Dataset, input_path, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prediction', nargs='?', type=Path)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data')
    parser.add_argument('--schema', type=Path, help='Output schema; defaults to the dataset-declared schema')
    args = parser.parse_args()
    contract = Contract(input_path(args.dataset, 'schema', args.schema))
    if args.prediction:
        contract.validate(read(args.prediction))
        print(f'Valid model output: {args.prediction}')
    else:
        dataset = Dataset(args.dataset, contract)
        print(f'Validated {len(dataset.samples)} image/annotation pairs and their SHA-256 hashes.')


if __name__ == '__main__':
    main()
