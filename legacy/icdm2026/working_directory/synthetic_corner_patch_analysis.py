import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from PIL import Image

DATASET_DIR = Path(r"E:\小龙虾\synthetic_corner_patch")
META_PATH = DATASET_DIR / "metadata.csv"
OUT_PATH = Path(r"E:\小龙虾\synthetic_corner_patch_quantitative_results.json")

EXPECTED_MAP = {
    "airplane": "top_left",
    "automobile": "top_right",
    "bird": "bottom_left",
    "cat": "bottom_right",
}
CORNERS = ["top_left", "top_right", "bottom_left", "bottom_right"]
TARGET_SPLITS = ["train", "val", "test_id", "test_ood"]


def entropy_from_counts(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def mutual_information(x, y):
    n = len(x)
    x_counts = Counter(x)
    y_counts = Counter(y)
    xy_counts = Counter(zip(x, y))
    mi = 0.0
    for (xi, yi), c in xy_counts.items():
        pxy = c / n
        px = x_counts[xi] / n
        py = y_counts[yi] / n
        mi += pxy * math.log2(pxy / (px * py))
    return mi


def normalized_mutual_information(x, y):
    mi = mutual_information(x, y)
    hx = entropy_from_counts(Counter(x).values())
    hy = entropy_from_counts(Counter(y).values())
    if hx == 0 or hy == 0:
        return 0.0
    return mi / math.sqrt(hx * hy)


def chi_square_and_cramers_v(x, y, x_levels=None, y_levels=None):
    if x_levels is None:
        x_levels = list(dict.fromkeys(x))
    if y_levels is None:
        y_levels = list(dict.fromkeys(y))
    table = pd.crosstab(pd.Series(x, name='x'), pd.Series(y, name='y'))
    table = table.reindex(index=x_levels, columns=y_levels, fill_value=0)
    obs = table.values
    n = obs.sum()
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    valid = expected != 0
    chi2 = float((((obs - expected) ** 2) / expected)[valid].sum())
    r, k = obs.shape
    denom = n * max(min(r - 1, k - 1), 1)
    v = math.sqrt(chi2 / denom) if denom > 0 else 0.0
    return chi2, v, table


def simple_cramers_v_from_table(table):
    obs = table.values
    n = obs.sum()
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    chi2 = float((((obs - expected) ** 2) / expected)[expected != 0].sum())
    r, k = obs.shape
    denom = n * max(min(r - 1, k - 1), 1)
    return chi2, math.sqrt(chi2 / denom) if denom > 0 else 0.0


def deterministic_predictability(df_split):
    per_corner = {}
    correct = 0
    for corner, group in df_split.groupby('patch_corner'):
        majority_label = group['class_name'].value_counts().idxmax()
        majority_count = int(group['class_name'].value_counts().max())
        total = int(len(group))
        acc = majority_count / total
        correct += majority_count
        per_corner[corner] = {
            'majority_label': majority_label,
            'majority_count': majority_count,
            'total': total,
            'majority_accuracy': acc,
            'label_distribution': {k: int(v) for k, v in group['class_name'].value_counts().to_dict().items()}
        }
    overall_acc = correct / len(df_split)
    return overall_acc, per_corner


def patch_detect_metrics(sample_paths):
    metrics = []
    for p in sample_paths:
        img = Image.open(p).convert('RGB')
        w, h = img.size
        coords = {
            'top_left': (0, 0),
            'top_right': (w-4, 0),
            'bottom_left': (0, h-4),
            'bottom_right': (w-4, h-4),
        }
        sums = {}
        is_red = {}
        for corner, (x0, y0) in coords.items():
            patch = [img.getpixel((x, y)) for y in range(y0, y0+4) for x in range(x0, x0+4)]
            red_count = sum(1 for px in patch if px == (255, 0, 0))
            sums[corner] = red_count
            is_red[corner] = red_count / 16.0
        best_corner = max(sums, key=sums.get)
        metrics.append({
            'file': str(p),
            'best_corner': best_corner,
            'corner_red_fraction': is_red,
            'best_corner_red_fraction': is_red[best_corner],
        })
    best_fracs = [m['best_corner_red_fraction'] for m in metrics]
    return {
        'sample_count': len(metrics),
        'best_corner_red_fraction_mean': sum(best_fracs)/len(best_fracs),
        'best_corner_red_fraction_min': min(best_fracs),
        'best_corner_red_fraction_max': max(best_fracs),
        'samples': metrics[:12],
    }


def main():
    df = pd.read_csv(META_PATH)
    df['expected_corner_from_map'] = df['class_name'].map(EXPECTED_MAP)
    df['expected_match_check'] = (df['expected_corner'] == df['expected_corner_from_map'])

    results = {
        'dataset_name': 'synthetic_corner_patch',
        'n_total': int(len(df)),
        'label_space': sorted(df['class_name'].unique().tolist()),
        'corner_space': CORNERS,
        'global_checks': {
            'all_expected_corner_consistent_with_identify_agent_mapping': bool(df['expected_match_check'].all()),
            'patch_size_unique': sorted(df['patch_size'].unique().tolist()),
            'patch_color_unique': sorted(df['patch_color'].unique().tolist()),
            'align_prob_unique': sorted(df['align_prob'].unique().tolist()),
        },
        'split_metrics': {},
    }

    id_df = df[df['split'].isin(['train', 'val', 'test_id'])].copy()
    ood_df = df[df['split'] == 'test_ood'].copy()

    for split in TARGET_SPLITS:
        d = df[df['split'] == split].copy()
        n = len(d)
        corner_counts = {k: int(v) for k, v in d['patch_corner'].value_counts().reindex(CORNERS, fill_value=0).to_dict().items()}
        expected_counts = {k: int(v) for k, v in d['expected_corner'].value_counts().reindex(CORNERS, fill_value=0).to_dict().items()}
        label_counts = {k: int(v) for k, v in d['class_name'].value_counts().to_dict().items()}

        mi = mutual_information(d['patch_corner'].tolist(), d['class_name'].tolist())
        nmi = normalized_mutual_information(d['patch_corner'].tolist(), d['class_name'].tolist())
        chi2, cramers_v, contingency = chi_square_and_cramers_v(d['patch_corner'].tolist(), d['class_name'].tolist(), CORNERS, sorted(EXPECTED_MAP.keys()))
        predict_acc, per_corner = deterministic_predictability(d)

        per_label_alignment = {}
        for label, g in d.groupby('class_name'):
            aligned = int((g['patch_corner'] == g['expected_corner']).sum())
            total = int(len(g))
            per_label_alignment[label] = {
                'expected_corner': g['expected_corner'].iloc[0],
                'aligned_count': aligned,
                'total': total,
                'aligned_rate': aligned / total,
            }

        results['split_metrics'][split] = {
            'n_samples': int(n),
            'patch_corner_distribution': corner_counts,
            'expected_corner_distribution': expected_counts,
            'label_distribution': label_counts,
            'patch_expected_alignment': {
                'aligned_count': int((d['patch_corner'] == d['expected_corner']).sum()),
                'aligned_rate': float((d['patch_corner'] == d['expected_corner']).mean()),
                'misaligned_count': int((d['patch_corner'] != d['expected_corner']).sum()),
            },
            'per_label_expected_alignment': per_label_alignment,
            'patch_corner_vs_label_predictability': {
                'mutual_information_bits': mi,
                'normalized_mutual_information': nmi,
                'chi_square': chi2,
                'cramers_v': cramers_v,
                'majority_decoding_accuracy_from_patch_corner': predict_acc,
                'per_corner_majority_label': per_corner,
                'contingency_table_patch_corner_x_label': {
                    row: {col: int(contingency.loc[row, col]) for col in contingency.columns}
                    for row in contingency.index
                },
            },
        }

    id_mi = mutual_information(id_df['patch_corner'].tolist(), id_df['class_name'].tolist())
    id_nmi = normalized_mutual_information(id_df['patch_corner'].tolist(), id_df['class_name'].tolist())
    id_chi2, id_v, _ = chi_square_and_cramers_v(id_df['patch_corner'].tolist(), id_df['class_name'].tolist(), CORNERS, sorted(EXPECTED_MAP.keys()))
    id_pred, _ = deterministic_predictability(id_df)

    ood_mi = mutual_information(ood_df['patch_corner'].tolist(), ood_df['class_name'].tolist())
    ood_nmi = normalized_mutual_information(ood_df['patch_corner'].tolist(), ood_df['class_name'].tolist())
    ood_chi2, ood_v, _ = chi_square_and_cramers_v(ood_df['patch_corner'].tolist(), ood_df['class_name'].tolist(), CORNERS, sorted(EXPECTED_MAP.keys()))
    ood_pred, _ = deterministic_predictability(ood_df)

    id_align = float((id_df['patch_corner'] == id_df['expected_corner']).mean())
    ood_align = float((ood_df['patch_corner'] == ood_df['expected_corner']).mean())

    # Patch color separability: verify corner-localized pure red detectability.
    sample_paths = [DATASET_DIR / rp for rp in df['relative_path'].sample(min(64, len(df)), random_state=42).tolist()]
    patch_metrics = patch_detect_metrics(sample_paths)

    results['shortcut_metrics'] = {
        'corner-position shortcut': {
            'label_to_expected_corner_mapping': EXPECTED_MAP,
            'split_alignment_rates': {
                split: results['split_metrics'][split]['patch_expected_alignment']['aligned_rate']
                for split in TARGET_SPLITS
            },
            'split_alignment_counts': {
                split: {
                    'aligned_count': results['split_metrics'][split]['patch_expected_alignment']['aligned_count'],
                    'n_samples': results['split_metrics'][split]['n_samples'],
                }
                for split in TARGET_SPLITS
            },
            'predictability_by_split': {
                split: {
                    'mutual_information_bits': results['split_metrics'][split]['patch_corner_vs_label_predictability']['mutual_information_bits'],
                    'normalized_mutual_information': results['split_metrics'][split]['patch_corner_vs_label_predictability']['normalized_mutual_information'],
                    'cramers_v': results['split_metrics'][split]['patch_corner_vs_label_predictability']['cramers_v'],
                    'majority_decoding_accuracy_from_patch_corner': results['split_metrics'][split]['patch_corner_vs_label_predictability']['majority_decoding_accuracy_from_patch_corner'],
                }
                for split in TARGET_SPLITS
            },
        },
        'red-patch local artifact shortcut': {
            'patch_size': 4,
            'patch_color': '(255, 0, 0)',
            'metadata_uniformity': {
                'single_patch_size': len(df['patch_size'].unique()) == 1,
                'single_patch_color': len(df['patch_color'].unique()) == 1,
                'patch_presence_rate_from_metadata': 1.0,
            },
            'corner_local_red_separability': patch_metrics,
        },
        'distribution-shift vulnerability': {
            'id_splits': ['train', 'val', 'test_id'],
            'ood_splits': ['test_ood'],
            'alignment_rate_id': id_align,
            'alignment_rate_ood': ood_align,
            'alignment_rate_drop_abs': id_align - ood_align,
            'predictability_change': {
                'mutual_information_bits': {'id': id_mi, 'ood': ood_mi, 'abs_drop': id_mi - ood_mi},
                'normalized_mutual_information': {'id': id_nmi, 'ood': ood_nmi, 'abs_drop': id_nmi - ood_nmi},
                'cramers_v': {'id': id_v, 'ood': ood_v, 'abs_drop': id_v - ood_v},
                'majority_decoding_accuracy_from_patch_corner': {'id': id_pred, 'ood': ood_pred, 'abs_drop': id_pred - ood_pred},
                'chi_square': {'id': id_chi2, 'ood': ood_chi2, 'abs_drop': id_chi2 - ood_chi2},
            },
        },
    }

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        'out_path': str(OUT_PATH),
        'dataset_name': results['dataset_name'],
        'n_total': results['n_total'],
        'split_alignment_rates': results['shortcut_metrics']['corner-position shortcut']['split_alignment_rates'],
        'id_vs_ood_cramers_v': results['shortcut_metrics']['distribution-shift vulnerability']['predictability_change']['cramers_v'],
        'id_vs_ood_nmi': results['shortcut_metrics']['distribution-shift vulnerability']['predictability_change']['normalized_mutual_information'],
        'id_vs_ood_majority_decoding_accuracy': results['shortcut_metrics']['distribution-shift vulnerability']['predictability_change']['majority_decoding_accuracy_from_patch_corner'],
        'red_patch_corner_local_red_fraction_mean': results['shortcut_metrics']['red-patch local artifact shortcut']['corner_local_red_separability']['best_corner_red_fraction_mean'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
