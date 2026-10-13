"""Validate externally supplied pairs without loading any construction truth."""
import hashlib
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from shortcutminer.inspection.image import load_rgb_image_file


def allowed_mask(rgb, box):
    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    x0, y0 = round(box.x0*width), round(box.y0*height)
    x1, y1 = round(box.x1*width), round(box.y1*height)
    if x0 >= x1 or y0 >= y1:
        raise ValueError('Pair region has no pixel area')
    mask[y0:y1, x0:x1] = True
    return mask


@lru_cache(maxsize=8)
def _index(data):
    payload = json.loads(data)
    rows = payload['pairs']
    keys = [r['original_reference'] for r in rows]
    targets = [r['paired_reference'] for r in rows]
    if len(set(keys)) != len(keys) or len(set(targets)) != len(targets):
        raise ValueError('Provided pairs must be one-to-one')
    return dict(zip(keys, rows))


def resolve_pair(root, source_path, spec):
    if root is None:
        raise ValueError('Provided pairs require an approved dataset root')
    root = Path(root).resolve(strict=True)
    source = Path(source_path).resolve(strict=True)
    source_reference = source.relative_to(root).as_posix()
    index_path = (root / spec.pair_index_file).resolve(strict=True)
    index_path.relative_to(root)
    if index_path.is_symlink():
        raise ValueError('Pair index symlink rejected')
    data = index_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != spec.pair_index_sha256:
        raise ValueError('Provided pair index identity changed')
    row = _index(data).get(source_reference)
    if row is None:
        raise ValueError('No unique authorized pair for original image')
    target = (root / row['paired_reference']).resolve(strict=True)
    target.relative_to(root)
    if target.is_symlink() or source.is_symlink():
        raise ValueError('Pair image symlink rejected')
    if row['original_label'] != row['paired_label']:
        raise ValueError('Pair labels differ')
    for path, key in ((source, 'original_sha256'), (target, 'paired_sha256')):
        if hashlib.sha256(path.read_bytes()).hexdigest() != row[key]:
            raise ValueError('Provided pair image identity changed')
    original = load_rgb_image_file(source).rgb
    paired = load_rgb_image_file(target).rgb
    if original.shape != paired.shape:
        raise ValueError('Pair dimensions differ')
    mask = allowed_mask(original, spec.changed_region)
    if mask.all() or not np.array_equal(original[~mask], paired[~mask]):
        raise ValueError('Provided pair changes pixels outside proposed region')
    if not np.any(original[mask] != paired[mask]):
        raise ValueError('Provided pair makes no pixel change')
    return paired, row
