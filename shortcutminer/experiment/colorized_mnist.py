"""Read-only Colorized MNIST adapter with content-grouped split isolation."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

from shortcutminer.experiment.waterbirds_paper import sha, identity, write

PALETTE = {'red': [255, 0, 0], 'green': [0, 255, 0], 'blue': [0, 0, 255]}


def inspect_source(source):
    source = Path(source).resolve(strict=True)
    records = []
    for split in ['training', 'testing']:
        for path in sorted((source / split).glob('*/*')):
            if not path.is_file():
                raise RuntimeError('Unexpected non-file in class directory')
            path.resolve(strict=True).relative_to(source)
            label, color = path.parent.name, path.stem.split('_')[0]
            if label not in {str(i) for i in range(10)} or color not in PALETTE:
                raise RuntimeError('Unsupported digit label or color encoding')
            with Image.open(path) as image:
                if image.mode != 'RGB' or image.size != (28, 28):
                    raise RuntimeError('Expected canonical RGB 28x28 images')
                pixels = np.array(image)
            intensity = pixels.min(axis=2)
            if not np.array_equal(pixels, np.maximum(intensity[..., None], np.array(PALETTE[color], dtype=np.uint8))):
                raise RuntimeError('Pixels disagree with declared color composition')
            records.append({'reference': path.relative_to(source).as_posix(), 'label': label,
                'color': color, 'original_split': split, 'file_sha256': sha(path),
                'digit_intensity_sha256': hashlib.sha256(intensity.tobytes()).hexdigest()})
    if not records or any(not any(r['original_split'] == s for r in records) for s in ['training', 'testing']):
        raise RuntimeError('Both original partitions must be present')
    return records


def partition(records, protocol):
    test = [r for r in records if r['original_split'] == 'testing']
    test_content = {r['digit_intensity_sha256'] for r in test}
    excluded = [r for r in records if r['original_split'] == 'training' and r['digit_intensity_sha256'] in test_content]
    eligible = [r for r in records if r['original_split'] == 'training' and r['digit_intensity_sha256'] not in test_content]
    if len(eligible) < 100:
        raise RuntimeError('Too few eligible training images')
    groups = [r['digit_intensity_sha256'] for r in eligible]
    remaining, id_positions = next(GroupShuffleSplit(n_splits=1, test_size=protocol['id_fraction'], random_state=protocol['seed']).split(eligible, groups=groups))
    remaining_rows = [eligible[i] for i in remaining]
    train_local, val_local = next(GroupShuffleSplit(n_splits=1, test_size=protocol['validation_fraction_of_remaining'], random_state=protocol['seed'] + 1).split(
        remaining_rows, groups=[r['digit_intensity_sha256'] for r in remaining_rows]))
    assignments = {eligible[i]['reference']: 'id' for i in id_positions}
    assignments.update({remaining_rows[i]['reference']: 'train' for i in train_local})
    assignments.update({remaining_rows[i]['reference']: 'val' for i in val_local})
    assignments.update({r['reference']: 'test' for r in test})
    sets = {s: {r['digit_intensity_sha256'] for r in records if assignments.get(r['reference']) == s} for s in ['train', 'val', 'id', 'test']}
    for i, s in enumerate(sets):
        for t in list(sets)[i + 1:]:
            if sets[s] & sets[t]:
                raise RuntimeError('Digit intensity content leaks across partitions')
    return assignments, excluded


def prepare(config):
    records = inspect_source(config['dataset_root'])
    source_hash = identity(records)
    if source_hash != config['expected_source_inventory_sha256']:
        raise RuntimeError('Source dataset changed from frozen inventory')
    assignments, excluded = partition(records, config['split_protocol'])
    root = Path(config['derived_dataset_root']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = identity({'source_inventory_sha256': source_hash, 'split_protocol': config['split_protocol']})
    existing_path = root / 'input_manifest.json'
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text(encoding='utf-8'))
        if existing['input_hash'] != fingerprint:
            raise RuntimeError('Existing input view belongs to a different protocol')
        for reference, expected in existing['view_hashes'].items():
            if sha(root / reference) != expected:
                raise RuntimeError('Frozen input metadata changed')
        for r in records:
            if sha(root / 'images' / r['reference']) != r['file_sha256']:
                raise RuntimeError('Frozen input image changed')
        return root, existing
    source = Path(config['dataset_root']).resolve(strict=True)
    rows, mapping = [], []
    for r in records:
        origin = source / r['reference']
        target = root / 'images' / r['reference']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(origin, target)
        if sha(target) != r['file_sha256']:
            raise RuntimeError('Derived image identity changed')
        selected_split = assignments.get(r['reference'], 'excluded_overlap')
        mapping.append({**r, 'execution_split': selected_split})
        if selected_split != 'excluded_overlap':
            rows.append({'img_id': hashlib.sha256(r['reference'].encode()).hexdigest()[:24],
                'img_filename': 'images/' + r['reference'], 'y': r['label'],
                'split': selected_split, 'color': r['color']})
    # Stable shuffled row order avoids class-sorted truncation in bounded tools.
    order = np.random.default_rng(config['split_protocol']['seed']).permutation(len(rows))
    rows = [rows[i] for i in order]
    for name, selected in [('model_metadata.csv', rows), ('discovery_metadata.csv', [r for r in rows if r['split'] == 'train'])]:
        with (root / name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['img_id', 'img_filename', 'y', 'split', 'color'])
            writer.writeheader()
            writer.writerows(selected)
    write(root / 'partition_map.json', mapping)
    counts = {s: sum(r['split'] == s for r in rows) for s in ['train', 'val', 'id', 'test']}
    for s in counts:
        if {r['y'] for r in rows if r['split'] == s} != {str(i) for i in range(10)}:
            raise RuntimeError('Partition does not contain all digit labels')
    if counts['id'] > 1000:
        raise RuntimeError('ID exceeds the current complete-paired intervention bound')
    manifest = {'input_hash': fingerprint, 'data_sha256': source_hash, 'partition_counts': counts,
        'view_hashes': {f: sha(root / f) for f in ['model_metadata.csv', 'discovery_metadata.csv', 'partition_map.json']},
        'split_protocol': config['split_protocol'], 'source_dataset': str(source),
        'source_image_count': len(records), 'source_inventory_sha256': source_hash,
        'excluded_training_test_content_overlap': excluded,
        'split_isolation': 'grouped by exact recovered digit intensity SHA256; original test retained',
        'discovery_mode': 'metadata_assisted_train_partition_only',
        'image_strategy': 'hard links to verified original bytes; original data read-only',
        'color_encoding': 'filename color verified against every original RGB pixel'}
    write(existing_path, manifest)
    return root, manifest
