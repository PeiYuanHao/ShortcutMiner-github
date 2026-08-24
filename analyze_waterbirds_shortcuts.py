import os, json, math, random, statistics
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATASET_ROOT = r"E:\小龙虾\waterbirds"
OUTPUT_JSON = r"E:\小龙虾\waterbirds_quantitative_analysis.json"
MAX_PER_CLASS = 20
BORDER_FRAC = 0.12

WATER_KEYWORDS = {
    'albatross','auklet','booby','cormorant','crab','duck','egret','frigatebird','fulmar','gadwall',
    'grebe','gull','jaeger','kingfisher','kittiwake','loon','merganser','murre','pelican','petrel',
    'puffin','sandpiper','seaduck','shearwater','tern','waterthrush','willet','anser','goose','swan',
    'coot','loon','turnstone','oystercatcher','avocet'
}

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def safe_float(x):
    if x is None:
        return None
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    return statistics.pstdev(xs)


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def quantile(xs, q):
    xs = sorted([x for x in xs if x is not None])
    if not xs:
        return None
    idx = (len(xs) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return xs[lo]
    frac = idx - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def auc_from_scores(labels, scores):
    pairs = [(l, s) for l, s in zip(labels, scores) if s is not None]
    if not pairs:
        return None
    labels = np.array([p[0] for p in pairs], dtype=int)
    scores = np.array([p[1] for p in pairs], dtype=float)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # Mann-Whitney U based AUC with tie handling
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    sorted_scores = scores[order]
    n = len(scores)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[order[i:j+1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[labels == 1].sum()
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def cohens_d(a, b):
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = mean(a), mean(b)
    va = np.var(a)
    vb = np.var(b)
    pooled = ((len(a)-1)*va + (len(b)-1)*vb) / (len(a)+len(b)-2)
    if pooled <= 0:
        return 0.0
    return float((ma - mb) / math.sqrt(pooled))


def eta_squared(groups):
    # groups: dict[name] -> list[float]
    clean = {k: [x for x in v if x is not None] for k, v in groups.items()}
    clean = {k: v for k, v in clean.items() if len(v) > 0}
    if len(clean) < 2:
        return None
    all_vals = [x for vals in clean.values() for x in vals]
    grand = mean(all_vals)
    ss_between = sum(len(vals) * ((mean(vals) - grand) ** 2) for vals in clean.values())
    ss_total = sum((x - grand) ** 2 for x in all_vals)
    if ss_total <= 0:
        return 0.0
    return float(ss_between / ss_total)


def class_name_from_folder(folder_name):
    parts = folder_name.split('.', 1)
    raw = parts[1] if len(parts) == 2 else folder_name
    return raw.replace('_', ' ')


def infer_habitat_cluster(class_name):
    text = class_name.lower().replace('-', ' ')
    for kw in WATER_KEYWORDS:
        if kw in text:
            return 'water_associated'
    return 'land_associated'


def list_class_folders(root):
    folders = []
    for p in sorted(Path(root).iterdir()):
        if p.is_dir():
            files = [x for x in p.rglob('*') if x.is_file() and x.suffix.lower() in IMG_EXTS]
            if files:
                folders.append((p, files))
    return folders


def sample_files(files, k):
    files = sorted(files)
    if len(files) <= k:
        return files
    rng = random.Random(SEED + len(files))
    return sorted(rng.sample(files, k))


def rgb_to_hsv_arr(rgb):
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.max(arr, axis=-1)
    mn = np.min(arr, axis=-1)
    diff = mx - mn
    s = np.where(mx == 0, 0, diff / np.maximum(mx, 1e-8))
    return s


def image_metrics(path):
    try:
        img = Image.open(path).convert('RGB')
        w, h = img.size
        arr = np.array(img)
        bh = max(1, int(h * BORDER_FRAC))
        bw = max(1, int(w * BORDER_FRAC))
        mask = np.zeros((h, w), dtype=bool)
        mask[:bh, :] = True
        mask[-bh:, :] = True
        mask[:, :bw] = True
        mask[:, -bw:] = True
        border = arr[mask]
        center = arr[~mask] if (~mask).sum() > 0 else arr.reshape(-1, 3)

        gray = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        sharpness = float(np.var(gx) + np.var(gy))

        border_blue = float(border[:, 2].mean() / 255.0)
        border_green = float(border[:, 1].mean() / 255.0)
        border_red = float(border[:, 0].mean() / 255.0)
        border_b_minus_g = border_blue - border_green
        border_b_plus_g = border_blue + border_green
        border_sat = float(rgb_to_hsv_arr(border.reshape(-1, 1, 3)).mean())
        center_sat = float(rgb_to_hsv_arr(center.reshape(-1, 1, 3)).mean())
        brightness = float(gray.mean() / 255.0)
        aspect_ratio = float(w / h) if h else None
        megapixels = float((w * h) / 1_000_000.0)

        return {
            'width': w,
            'height': h,
            'aspect_ratio': aspect_ratio,
            'megapixels': megapixels,
            'brightness': brightness,
            'sharpness': sharpness,
            'border_red': border_red,
            'border_green': border_green,
            'border_blue': border_blue,
            'border_b_minus_g': border_b_minus_g,
            'border_b_plus_g': border_b_plus_g,
            'border_saturation': border_sat,
            'center_saturation': center_sat,
        }
    except Exception as e:
        return {'error': str(e)}


def summarize_metric(values):
    vals = [v for v in values if v is not None]
    return {
        'mean': mean(vals),
        'std': std(vals),
        'median': median(vals),
        'q10': quantile(vals, 0.10),
        'q90': quantile(vals, 0.90),
        'min': min(vals) if vals else None,
        'max': max(vals) if vals else None,
        'n': len(vals),
    }


def main():
    root = Path(DATASET_ROOT)
    if not root.exists():
        raise FileNotFoundError(DATASET_ROOT)

    class_folders = list_class_folders(root)
    records = []
    class_counts = {}
    inferred_clusters = {}

    for folder, files in class_folders:
        class_folder = folder.name
        class_name = class_name_from_folder(class_folder)
        class_counts[class_name] = len(files)
        cluster = infer_habitat_cluster(class_name)
        inferred_clusters[class_name] = cluster
        for fp in sample_files(files, MAX_PER_CLASS):
            m = image_metrics(str(fp))
            rec = {
                'class_folder': class_folder,
                'class_name': class_name,
                'image_path': str(fp),
                'inferred_habitat_cluster': cluster,
            }
            rec.update(m)
            records.append(rec)

    ok_records = [r for r in records if 'error' not in r]
    metric_names = [
        'border_blue','border_green','border_red','border_b_minus_g','border_b_plus_g',
        'border_saturation','center_saturation','brightness','aspect_ratio','megapixels','sharpness'
    ]

    by_class = defaultdict(list)
    for r in ok_records:
        by_class[r['class_name']].append(r)

    by_cluster = defaultdict(list)
    for r in ok_records:
        by_cluster[r['inferred_habitat_cluster']].append(r)

    cluster_labels = [1 if r['inferred_habitat_cluster'] == 'water_associated' else 0 for r in ok_records]

    cluster_metric_stats = {}
    separability = {}
    for metric in metric_names:
        water_vals = [r[metric] for r in by_cluster['water_associated']]
        land_vals = [r[metric] for r in by_cluster['land_associated']]
        cluster_metric_stats[metric] = {
            'water_associated': summarize_metric(water_vals),
            'land_associated': summarize_metric(land_vals),
            'difference_water_minus_land': None if mean(water_vals) is None or mean(land_vals) is None else mean(water_vals) - mean(land_vals),
            'cohens_d_water_minus_land': cohens_d(water_vals, land_vals),
            'auc_water_vs_land': auc_from_scores(cluster_labels, [r[metric] for r in ok_records]),
        }
        class_groups = {cls: [x[metric] for x in recs] for cls, recs in by_class.items()}
        separability[metric] = {
            'eta_squared_by_200_classes': eta_squared(class_groups)
        }

    # class-level correlations between counts and style / background metrics
    class_level = {}
    class_rows = []
    for cls, recs in by_class.items():
        row = {'class_name': cls, 'count': class_counts[cls], 'cluster': inferred_clusters[cls]}
        for metric in metric_names:
            row[metric] = mean([r[metric] for r in recs])
        class_rows.append(row)

    def pearson(xs, ys):
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if len(pairs) < 2:
            return None
        x = np.array([p[0] for p in pairs], dtype=float)
        y = np.array([p[1] for p in pairs], dtype=float)
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    count_values = [r['count'] for r in class_rows]
    count_metric_corr = {}
    for metric in metric_names:
        count_metric_corr[metric] = pearson(count_values, [r[metric] for r in class_rows])

    counts = list(class_counts.values())
    total_images = sum(counts)
    n_classes = len(counts)
    imbalance = {
        'num_classes': n_classes,
        'total_images': total_images,
        'min_per_class': min(counts) if counts else None,
        'max_per_class': max(counts) if counts else None,
        'mean_per_class': mean(counts),
        'median_per_class': median(counts),
        'std_per_class': std(counts),
        'max_min_ratio': (max(counts) / min(counts)) if counts and min(counts) > 0 else None,
        'p90_p10_ratio': (quantile(counts, 0.90) / quantile(counts, 0.10)) if counts and quantile(counts, 0.10) not in (0, None) else None,
        'coefficient_of_variation': (std(counts) / mean(counts)) if counts and mean(counts) else None,
        'top_10_classes_by_count': sorted(class_counts.items(), key=lambda x: (-x[1], x[0]))[:10],
        'bottom_10_classes_by_count': sorted(class_counts.items(), key=lambda x: (x[1], x[0]))[:10],
    }

    summary = {
        'dataset_name': 'waterbirds',
        'dataset_root': DATASET_ROOT,
        'analysis_scope': {
            'class_folders_detected': len(class_folders),
            'sampled_images_for_metric_analysis': len(ok_records),
            'max_sample_per_class': MAX_PER_CLASS,
            'official_group_labels_available': False,
            'surrogate_grouping_used': '基于类别名关键词的 habitat proxy（二分为 water_associated / land_associated）',
            'surrogate_group_counts': dict(Counter(inferred_clusters.values())),
        },
        'shortcuts': {
            '背景-类别伪相关': {
                'key_metrics': {
                    'border_blue': cluster_metric_stats['border_blue'],
                    'border_green': cluster_metric_stats['border_green'],
                    'border_b_minus_g': cluster_metric_stats['border_b_minus_g'],
                    'border_saturation': cluster_metric_stats['border_saturation'],
                    'class_separability_eta2': {
                        k: separability[k]['eta_squared_by_200_classes']
                        for k in ['border_blue','border_green','border_b_minus_g','border_saturation','brightness']
                    }
                }
            },
            '栖息地共现捷径': {
                'proxy_definition': '用类别名推断栖息地簇；以边缘蓝/绿通道、蓝减绿、亮度/饱和度作为水域/植被场景代理',
                'key_metrics': {
                    k: cluster_metric_stats[k] for k in ['border_blue','border_green','border_b_minus_g','brightness','border_saturation']
                }
            },
            '场景颜色分布捷径': {
                'key_metrics': {
                    k: cluster_metric_stats[k] for k in ['border_red','border_green','border_blue','border_b_plus_g','center_saturation']
                }
            },
            '数据源/拍摄风格捷径': {
                'key_metrics': {
                    k: cluster_metric_stats[k] for k in ['aspect_ratio','megapixels','sharpness']
                },
                'class_separability_eta2': {
                    k: separability[k]['eta_squared_by_200_classes'] for k in ['aspect_ratio','megapixels','sharpness']
                }
            },
            '轻度类数量不均衡': imbalance,
        },
        'class_count_vs_style_correlation': count_metric_corr,
        'limitations': [
            '当前目录结构为 CUB 风格 200 类细粒度分类，不是官方 Waterbirds 二分类 benchmark。',
            '未发现官方 group metadata（如 landbird-on-water / waterbird-on-land 等组合标签），因此无法直接复现标准 group DRO 统计。',
            '栖息地共现分析采用类别名关键词生成的 surrogate habitat grouping，属于可替代代理方案，不等同于官方标注。',
            '背景颜色使用图像边缘区域近似，不能完全等价于精确分割后的背景。'
        ]
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
