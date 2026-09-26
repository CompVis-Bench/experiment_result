"""Portable dataset, output-contract and audit utilities."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"Non-JSON constant: {value}")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def read(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def input_path(dataset, name, override=None):
    """Resolve an explicit input, a dataset-declared input, or the legacy default."""
    if override is not None:
        return Path(override)
    dataset = Path(dataset).resolve()
    manifest = dataset / 'manifest.json'
    inputs = read(manifest).get('inputs', {}) if manifest.exists() else {}
    filename = {'schema': 'output.schema.json', 'prompt': 'prompt.txt', 'system': 'system.txt'}[name]
    if name in inputs:
        path = (dataset / inputs[name]).resolve()
        if not path.is_relative_to(dataset):
            raise ValueError(f'Dataset {name} path escapes its directory')
        return path
    return dataset / filename if (dataset / filename).is_file() else ROOT / 'contracts' / filename


def scoring_target(annotation, annotation_only_channels=None):
    """Project explicitly declared annotation-only channels without changing the source."""
    projection = annotation_only_channels or {}
    charts = []
    for chart in annotation['charts']:
        excluded = projection.get(chart['variation'], [])
        charts.append({**chart, 'encodings': {k: v for k, v in chart['encodings'].items() if k not in excluded}})
    return {'charts': charts}


class Contract:
    def __init__(self, schema_path):
        self.schema = read(schema_path)
        Draft202012Validator.check_schema(self.schema)
        self.validator = Draft202012Validator(self.schema)
        self.labels = {v["properties"]["variation"]["const"]
                       for v in self.schema["properties"]["charts"]["items"]["oneOf"]}

    def validate(self, payload):
        error = next(self.validator.iter_errors(payload), None)
        if error:
            raise ValueError(f"Schema violation at {list(error.absolute_path)}: {error.message[:400]}")
        charts = payload["charts"]
        ids = {c["chart_id"]: c for c in charts}
        if len(ids) != len(charts):
            raise ValueError("Duplicate chart_id")
        for c in charts:
            for target in c.get("link_targets", []):
                if target not in ids or target == c["chart_id"] or ids[target]["variation"] == "link":
                    raise ValueError("Dangling/self/link-to-link target")
        return payload


class Dataset:
    """Final images + one annotation per sample, verified through the manifest."""
    def __init__(self, path, contract):
        self.path = Path(path).resolve()
        raw = (self.path/'manifest.json').read_bytes()
        self.sha256 = sha(raw)
        self.manifest = strict_json(raw)
        self.samples = self.manifest['samples']
        self.annotation_only_channels = self.manifest.get('annotation_only_channels', {})
        if not isinstance(self.annotation_only_channels, dict) or any(
                not isinstance(channels, list) or not all(isinstance(ch, str) for ch in channels)
                for channels in self.annotation_only_channels.values()):
            raise ValueError('Invalid annotation-only channel declaration')
        if len(self.samples) != self.manifest['sample_count'] or len({r['sample_id'] for r in self.samples}) != len(self.samples):
            raise ValueError('Invalid sample manifest')
        for row in self.samples:
            sid = row['sample_id']
            if not re.fullmatch(r'[A-Za-z0-9_-]+', sid):
                raise ValueError('Unsafe sample ID')
            for key in ('image', 'annotation'):
                if sha(self.bytes(row[key])) != row[key + '_sha256']:
                    raise ValueError(f'Manifest checksum mismatch: {row[key]}')
            annotation = strict_json(self.bytes(row['annotation']))
            if annotation.get('sample_id', sid) != sid:
                raise ValueError(f'Annotation sample ID mismatch: {sid}')
            contract.validate(self.target(row))
            # This hash identifies the sole annotation used as the scoring target.
            row['target_sha256'] = row['annotation_sha256']

    def bytes(self, name):
        path = (self.path/name).resolve()
        if not path.is_relative_to(self.path):
            raise ValueError('Dataset path escapes data directory')
        return path.read_bytes()

    def target(self, row):
        return scoring_target(strict_json(self.bytes(row['annotation'])), self.annotation_only_channels)


def git_state():
    def call(*args):
        p = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    try:
        return dict(commit=call('rev-parse', 'HEAD'), status=call('status', '--porcelain'))
    except OSError:
        return dict(commit=None, status=None)


def implementation_hashes():
    paths = sorted(p for folder in ('scripts', 'runners', 'eval', 'tools') for p in (ROOT / folder).rglob('*')
                   if p.is_file() and p.suffix in ('.py', '.sh'))
    return {p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in paths}
