import json
import math
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(r"E:\小龙虾\waterbirds")
OUT = Path(r"E:\小龙虾\waterbirds_quantitative_results.json")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Proxy habitat grouping because official Waterbirds benchmark group labels are absent.
WATER_KEYWORDS = [
    'Albatross', 'Auklet', 'Cormorant', 'Frigatebird', 'Fulmar', 'Gadwall',
    'Grebe', 'Guillemot', 'Gull', 'Jaeger', 'Kittiwake', 'Loon', 'Mallard',
    'Merganser', 'Pelican', 'Puffin', 'Tern', 'Kingfisher', 'Waterthrush',
    'Marsh_Wren', 'Seaside_Sparrow', 'Nelson_Sharp_tailed_Sparrow',
    'Fish_Crow', 'Cormorant', 'Pelagic', 'Pacific_Loon', 'Pied_billed_Grebe',
    'Horned_Puffin', 'Pigeon_Guillemot'
]

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SAMPLE_PER_CLASS = 20  # align to upstream quick probe budget


def class_name_from_folder(folder_name: str) -> str:
    parts = folder_name.split('.', 1)
    return parts[1] if len(parts) == 2 else folder_name


def is_water_proxy(class_name: str) -> bool:
    return any(k in class_name for k in WATER_KEYWORDS)


def entropy_from_counts(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0
    out = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            out -= p * math.log2(p)
    return out


def mutual_information(x, y):
    n = len(x)
    xc = Counter(x)
    yc = Counter(y)
    xyc = Counter(zip(x, y))
    mi = 0.0
    for (xi, yi), c in xyc.items():
        pxy = c / n
        px = xc[xi] / n
        py = yc[yi] / n
        mi += pxy * math.log2(pxy / (px * py))
    return mi


def normalized_mutual_information(x, y):
    mi = mutual_information(x, y)
    hx = entropy_from_counts(Counter(x).values())
    hy = entropy_from_counts(Counter(y).values())
    if hx == 0 or hy == 0:
        return 0.0
    return mi / math.sqrt(hx * hy)


def cramers_v(x, y):
    x_levels = list(dict.fromkeys(x))
    y_levels = list(dict.fromkeys(y))
    xi = {k: i for i, k in enumerate(x_levels)}
    yi = {k: i for i, k in enumerate(y_levels)}
    obs = np.zeros((len(x_levels), len(y_levels)), dtype=np.float64)
    for a, b in zip(x, y):
        obs[xi[a], yi[b]] += 1
    n = obs.sum()
    row = obs.sum(axis=1, keepdims=True)
    col = obs.sum(axis=0, keepdims=True)
    exp = row @ col / n
    valid = exp != 0
    chi2 = float((((obs - exp) ** 2) / exp)[valid].sum())
    denom = n * max(min(obs.shape[0] - 1, obs.shape[1] - 1), 1)
    v = math.sqrt(chi2 / denom) if denom > 0 else 0.0
    return chi2, v


def eta_squared(groups, values):
    values = np.asarray(values, dtype=np.float64)
    grand_mean = values.mean()
    ss_between = 0.0
    ss_total = ((values - grand_mean) ** 2).sum()
    grouped = defaultdict(list)
    for g, v in zip(groups, values):
        grouped[g].append(v)
    for g, vals in grouped.items():
        vals = np.asarray(vals, dtype=np.float64)
        ss_between += len(vals) * (vals.mean() - grand_mean) ** 2
    if ss_total <= 1e-12:
        return 0.0
    return float(ss_between / ss_total)


def point_biserial(binary_groups, values):
    vals = np.asarray(values, dtype=np.float64)
    groups = np.asarray(binary_groups)
    g1 = vals[groups == 1]
    g0 = vals[groups == 0]
    if len(g1) == 0 or len(g0) == 0:
        return 0.0
    m1, m0 = g1.mean(), g0.mean()
    s = vals.std(ddof=0)
    if s == 0:
        return 0.0
    p = len(g1) / len(vals)
    q = len(g0) / len(vals)
    return float(((m1 - m0) / s) * math.sqrt(p * q))


def laplacian_variance(gray):
    arr = gray.astype(np.float32)
    center = arr[1:-1, 1:-1]
    lap = -4 * center + arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:]
    return float(np.var(lap))


def safe_open_image(path):
    with Image.open(path) as img:
        return img.convert('RGB')


def image_metrics(path: Path):
    img = safe_open_image(path)
    w, h = img.size
    arr = np.asarray(img)
    arrf = arr.astype(np.float32) / 255.0
    # Edge band = 10% border, min 4 px, max 32 px
    band_x = max(4, min(32, int(round(w * 0.10))))
    band_y = max(4, min(32, int(round(h * 0.10))))
    mask = np.zeros((h, w), dtype=bool)
    mask[:band_y, :] = True
    mask[-band_y:, :] = True
    mask[:, :band_x] = True
    mask[:, -band_x:] = True
    edge = arrf[mask]
    overall = arrf.reshape(-1, 3)
    border_rgb = edge.mean(axis=0)
    overall_rgb = overall.mean(axis=0)
    # HSV-ish simple proxies
    rgb_max = edge.max(axis=1)
    rgb_min = edge.min(axis=1)
    sat = np.where(rgb_max > 1e-8, (rgb_max - rgb_min) / np.maximum(rgb_max, 1e-8), 0.0)
    value = rgb_max
    blue_mask = (edge[:, 2] > edge[:, 1]) & (edge[:, 2] > edge[:, 0])
    green_mask = (edge[:, 1] > edge[:, 2]) & (edge[:, 1] > edge[:, 0])
    gray = np.asarray(img.convert('L'))
    sharp = laplacian_variance(gray)
    return {
        'width': int(w),
        'height': int(h),
        'aspect_ratio': float(w / h) if h else 0.0,
        'pixels': int(w * h),
        'border_red_mean': float(border_rgb[0]),
        'border_green_mean': float(border_rgb[1]),
        'border_blue_mean': float(border_rgb[2]),
        'overall_red_mean': float(overall_rgb[0]),
        'overall_green_mean': float(overall_rgb[1]),
        'overall_blue_mean': float(overall_rgb[2]),
        'border_dominant_blue_frac': float(blue_mask.mean()),
        'border_dominant_green_frac': float(green_mask.mean()),
        'border_saturation_mean': float(sat.mean()),
        'border_value_mean': float(value.mean()),
        'sharpness_laplacian_var': sharp,
        'file_size_bytes': int(path.stat().st_size),
    }


def summarize_numeric(records, key):
    vals = np.array([r[key] for r in records], dtype=np.float64)
    return {
        'mean': float(vals.mean()),
        'std': float(vals.std(ddof=0)),
        'min': float(vals.min()),
        'q25': float(np.quantile(vals, 0.25)),
        'median': float(np.quantile(vals, 0.5)),
        'q75': float(np.quantile(vals, 0.75)),
        'max': float(vals.max()),
    }


def main():
    class_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()])
    class_info = []
    all_records = []
    sampled_records = []
    sampled_paths = []

    for class_dir in class_dirs:
        cname = class_name_from_folder(class_dir.name)
        files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        habitat = 'water' if is_water_proxy(cname) else 'land'
        class_info.append({
            'folder': class_dir.name,
            'class_name': cname,
            'sample_count': len(files),
            'habitat_proxy': habitat,
        })
        for fp in files:
            all_records.append({
                'path': str(fp),
                'class_name': cname,
                'folder': class_dir.name,
                'habitat_proxy': habitat,
            })
        sample_n = min(SAMPLE_PER_CLASS, len(files))
        idxs = np.linspace(0, len(files) - 1, sample_n, dtype=int) if sample_n > 0 else []
        for idx in idxs:
            sampled_paths.append((cname, habitat, files[idx]))

    for cname, habitat, fp in sampled_paths:
        m = image_metrics(fp)
        m.update({'path': str(fp), 'class_name': cname, 'habitat_proxy': habitat})
        sampled_records.append(m)

    # Class count imbalance on full dataset
    count_values = np.array([c['sample_count'] for c in class_info], dtype=np.int64)
    total_images = int(count_values.sum())
    mean_count = float(count_values.mean())
    std_count = float(count_values.std(ddof=0))
    cv_count = float(std_count / mean_count)
    imbalance_ratio = float(count_values.max() / count_values.min())

    # Correlations using sampled visual/style records
    class_labels = [r['class_name'] for r in sampled_records]
    habitat_labels = [r['habitat_proxy'] for r in sampled_records]
    habitat_bin = np.array([1 if h == 'water' else 0 for h in habitat_labels], dtype=np.int64)

    visual_keys = [
        'border_blue_mean', 'border_green_mean', 'border_red_mean',
        'overall_blue_mean', 'overall_green_mean', 'overall_red_mean',
        'border_dominant_blue_frac', 'border_dominant_green_frac',
        'border_saturation_mean', 'border_value_mean',
        'width', 'height', 'aspect_ratio', 'pixels',
        'sharpness_laplacian_var', 'file_size_bytes'
    ]

    feature_stats = {}
    for key in visual_keys:
        vals = [r[key] for r in sampled_records]
        feature_stats[key] = {
            'global_summary': summarize_numeric(sampled_records, key),
            'eta_squared_vs_class': eta_squared(class_labels, vals),
            'point_biserial_vs_habitat_proxy': point_biserial(habitat_bin, vals),
            'water_mean': float(np.mean([r[key] for r in sampled_records if r['habitat_proxy'] == 'water'])),
            'land_mean': float(np.mean([r[key] for r in sampled_records if r['habitat_proxy'] == 'land'])),
            'water_minus_land': float(np.mean([r[key] for r in sampled_records if r['habitat_proxy'] == 'water']) - np.mean([r[key] for r in sampled_records if r['habitat_proxy'] == 'land'])),
        }

    # Background color proxy discretization for categorical association
    def bin4(v):
        if v < 0.25:
            return 'very_low'
        if v < 0.5:
            return 'low'
        if v < 0.75:
            return 'high'
        return 'very_high'

    blue_bins = [bin4(r['border_blue_mean']) for r in sampled_records]
    green_bins = [bin4(r['border_green_mean']) for r in sampled_records]
    dom_scene = []
    for r in sampled_records:
        if r['border_dominant_blue_frac'] >= r['border_dominant_green_frac'] and r['border_dominant_blue_frac'] >= 0.33:
            dom_scene.append('blue_dominant')
        elif r['border_dominant_green_frac'] > r['border_dominant_blue_frac'] and r['border_dominant_green_frac'] >= 0.33:
            dom_scene.append('green_dominant')
        else:
            dom_scene.append('other_mixed')

    blue_chi2, blue_v = cramers_v(blue_bins, class_labels)
    green_chi2, green_v = cramers_v(green_bins, class_labels)
    scene_chi2, scene_v = cramers_v(dom_scene, habitat_labels)

    # Per-class summaries for selected keys
    selected_keys = ['border_blue_mean', 'border_green_mean', 'aspect_ratio', 'pixels', 'sharpness_laplacian_var']
    per_class = {}
    by_class = defaultdict(list)
    for r in sampled_records:
        by_class[r['class_name']].append(r)
    for cname, rows in by_class.items():
        per_class[cname] = {
            'habitat_proxy': rows[0]['habitat_proxy'],
            'sampled_n': len(rows),
            **{k: float(np.mean([x[k] for x in rows])) for k in selected_keys}
        }

    water_classes = [c for c in class_info if c['habitat_proxy'] == 'water']
    land_classes = [c for c in class_info if c['habitat_proxy'] == 'land']

    results = {
        'dataset_name': 'waterbirds',
        'dataset_path': str(ROOT),
        'dataset_structure': {
            'structure_assessment': 'CUB-200-2011-style class folders; not official binary Waterbirds benchmark layout',
            'n_classes': len(class_info),
            'n_images_total': total_images,
            'official_group_metadata_present': False,
            'proxy_grouping_used': True,
            'proxy_grouping_definition': {
                'type': 'class-name habitat proxy',
                'water_keywords': WATER_KEYWORDS,
                'water_classes': len(water_classes),
                'land_classes': len(land_classes),
            },
        },
        'shortcut_metrics': {
            '背景-类别伪相关': {
                'sample_budget': {'sample_per_class': SAMPLE_PER_CLASS, 'sampled_images_total': len(sampled_records)},
                'border_blue_mean_vs_class': {
                    'eta_squared': feature_stats['border_blue_mean']['eta_squared_vs_class'],
                    'binned_cramers_v': blue_v,
                    'binned_chi_square': blue_chi2,
                },
                'border_green_mean_vs_class': {
                    'eta_squared': feature_stats['border_green_mean']['eta_squared_vs_class'],
                    'binned_cramers_v': green_v,
                    'binned_chi_square': green_chi2,
                },
                'class_proxy_group_means': {
                    'water_border_blue_mean': feature_stats['border_blue_mean']['water_mean'],
                    'land_border_blue_mean': feature_stats['border_blue_mean']['land_mean'],
                    'water_minus_land_blue': feature_stats['border_blue_mean']['water_minus_land'],
                    'water_border_green_mean': feature_stats['border_green_mean']['water_mean'],
                    'land_border_green_mean': feature_stats['border_green_mean']['land_mean'],
                    'water_minus_land_green': feature_stats['border_green_mean']['water_minus_land'],
                },
            },
            '栖息地共现捷径': {
                'proxy_group_counts': {
                    'water_classes': len(water_classes),
                    'land_classes': len(land_classes),
                    'water_samples_sampled': int(sum(1 for r in sampled_records if r['habitat_proxy'] == 'water')),
                    'land_samples_sampled': int(sum(1 for r in sampled_records if r['habitat_proxy'] == 'land')),
                },
                'scene_proxy_association': {
                    'dominant_scene_categories': dict(Counter(dom_scene)),
                    'cramers_v_scene_vs_habitat_proxy': scene_v,
                    'chi_square_scene_vs_habitat_proxy': scene_chi2,
                    'border_dominant_blue_frac': {
                        'point_biserial_vs_habitat_proxy': feature_stats['border_dominant_blue_frac']['point_biserial_vs_habitat_proxy'],
                        'water_mean': feature_stats['border_dominant_blue_frac']['water_mean'],
                        'land_mean': feature_stats['border_dominant_blue_frac']['land_mean'],
                    },
                    'border_dominant_green_frac': {
                        'point_biserial_vs_habitat_proxy': feature_stats['border_dominant_green_frac']['point_biserial_vs_habitat_proxy'],
                        'water_mean': feature_stats['border_dominant_green_frac']['water_mean'],
                        'land_mean': feature_stats['border_dominant_green_frac']['land_mean'],
                    },
                },
            },
            '场景颜色分布捷径': {
                'border_color_features': {
                    key: feature_stats[key] for key in [
                        'border_blue_mean', 'border_green_mean', 'border_red_mean',
                        'overall_blue_mean', 'overall_green_mean', 'overall_red_mean',
                        'border_saturation_mean', 'border_value_mean'
                    ]
                },
            },
            '数据源/拍摄风格捷径': {
                'style_features': {
                    key: feature_stats[key] for key in [
                        'width', 'height', 'aspect_ratio', 'pixels', 'sharpness_laplacian_var', 'file_size_bytes'
                    ]
                },
            },
            '轻度类数量不均衡': {
                'class_count_summary': {
                    'n_classes': len(class_info),
                    'n_images_total': total_images,
                    'min_per_class': int(count_values.min()),
                    'max_per_class': int(count_values.max()),
                    'mean_per_class': mean_count,
                    'std_per_class': std_count,
                    'cv_per_class': cv_count,
                    'imbalance_ratio_max_over_min': imbalance_ratio,
                    'q25': float(np.quantile(count_values, 0.25)),
                    'median': float(np.quantile(count_values, 0.5)),
                    'q75': float(np.quantile(count_values, 0.75)),
                },
                'top_10_smallest_classes': sorted(class_info, key=lambda x: x['sample_count'])[:10],
                'top_10_largest_classes': sorted(class_info, key=lambda x: x['sample_count'], reverse=True)[:10],
            },
        },
        'supporting_outputs': {
            'per_class_sampled_feature_means': per_class,
        },
        'limitations': [
            '数据目录缺少官方 Waterbirds benchmark 的 group / split / place / y 元数据。',
            '因此无法直接复现官方 group-level spuriosity 指标，只能使用基于类名与背景颜色的代理分组统计。',
            '背景与场景统计基于每类最多 20 张图的抽样，以控制计算量；分辨率与类样本数统计基于全量文件。',
            '“数据源/拍摄风格捷径”这里采用可观测代理变量：分辨率、像素数、宽高比、文件大小、边缘背景颜色、清晰度。'
        ]
    }

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    preview = {
        'dataset_name': results['dataset_name'],
        'n_classes': results['dataset_structure']['n_classes'],
        'n_images_total': results['dataset_structure']['n_images_total'],
        'proxy_water_classes': len(water_classes),
        'proxy_land_classes': len(land_classes),
        'sampled_images_total': len(sampled_records),
        'water_border_blue_mean': results['shortcut_metrics']['背景-类别伪相关']['class_proxy_group_means']['water_border_blue_mean'],
        'land_border_blue_mean': results['shortcut_metrics']['背景-类别伪相关']['class_proxy_group_means']['land_border_blue_mean'],
        'water_minus_land_blue': results['shortcut_metrics']['背景-类别伪相关']['class_proxy_group_means']['water_minus_land_blue'],
        'water_border_green_mean': results['shortcut_metrics']['背景-类别伪相关']['class_proxy_group_means']['water_border_green_mean'],
        'land_border_green_mean': results['shortcut_metrics']['背景-类别伪相关']['class_proxy_group_means']['land_border_green_mean'],
        'water_minus_land_green': results['shortcut_metrics']['背景-类别伪相关']['class_proxy_group_means']['water_minus_land_green'],
        'border_blue_eta2': results['shortcut_metrics']['背景-类别伪相关']['border_blue_mean_vs_class']['eta_squared'],
        'border_green_eta2': results['shortcut_metrics']['背景-类别伪相关']['border_green_mean_vs_class']['eta_squared'],
        'scene_vs_habitat_cramers_v': results['shortcut_metrics']['栖息地共现捷径']['scene_proxy_association']['cramers_v_scene_vs_habitat_proxy'],
        'aspect_ratio_eta2': results['shortcut_metrics']['数据源/拍摄风格捷径']['style_features']['aspect_ratio']['eta_squared_vs_class'],
        'sharpness_eta2': results['shortcut_metrics']['数据源/拍摄风格捷径']['style_features']['sharpness_laplacian_var']['eta_squared_vs_class'],
        'class_count_cv': results['shortcut_metrics']['轻度类数量不均衡']['class_count_summary']['cv_per_class'],
        'out_path': str(OUT),
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
