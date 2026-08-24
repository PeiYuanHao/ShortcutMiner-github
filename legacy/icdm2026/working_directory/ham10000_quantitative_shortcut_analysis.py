import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2_contingency, kruskal, spearmanr
except Exception:
    chi2_contingency = kruskal = spearmanr = None

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except Exception:
    LogisticRegression = None

BASE = Path(r"E:\小龙虾")
DATASET_DIR = BASE / "HAM10000"
SCREEN_DIR = BASE / "ham10000_novel_shortcut_screen_results"
OUT_DIR = BASE / "ham10000_quantitative_shortcut_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_PATH = SCREEN_DIR / "image_features.csv"
META_PATH = DATASET_DIR / "HAM10000_metadata.csv"

FEATURES = pd.read_csv(FEATURE_PATH)
META = pd.read_csv(META_PATH)
# Merge metadata controls while preserving dx/lesion_id from feature file.
meta_controls = META[["image_id", "dx_type", "age", "sex", "localization"]].copy()
df = FEATURES.merge(meta_controls, on="image_id", how="left")

LABELS = sorted(df["dx"].dropna().unique().tolist())
N = len(df)
baseline = df["dx"].value_counts().reindex(LABELS, fill_value=0)
baseline_pct = baseline / N

candidate_defs = {
    "high_saturation_mean_S": {
        "feature": "mean_S", "direction": "high", "targets": ["nv"], "top_fracs": [0.01, 0.05, 0.10, 0.20],
        "desc": "高饱和度图像 cue"
    },
    "high_brightness_mean_V": {
        "feature": "mean_V", "direction": "high", "targets": ["nv"], "top_fracs": [0.01, 0.05, 0.10, 0.20],
        "desc": "高亮度图像 cue"
    },
    "low_brightness_mean_V": {
        "feature": "mean_V", "direction": "low", "targets": ["mel", "bkl"], "top_fracs": [0.01, 0.05, 0.10, 0.20],
        "desc": "低亮度图像 cue"
    },
    "dark_border_frac": {
        "feature": "dark_border_frac", "direction": "high", "targets": ["mel", "bkl"], "top_fracs": [0.01, 0.05, 0.10, 0.20],
        "desc": "黑边 cue"
    },
    "dark_corner_frac": {
        "feature": "dark_corner_frac", "direction": "high", "targets": ["mel", "bkl"], "top_fracs": [0.01, 0.05, 0.10, 0.20],
        "desc": "暗角 cue"
    },
    "border_center_l2": {
        "feature": "border_center_l2", "direction": "high", "targets": ["mel", "nv", "bkl", "bcc"], "top_fracs": [0.01, 0.05, 0.10, 0.20],
        "desc": "边缘-中心颜色差 cue"
    },
    "low_file_size": {
        "feature": "file_size", "direction": "low", "targets": ["bcc"], "top_fracs": [0.01, 0.05, 0.10, 0.20],
        "desc": "低 JPEG 文件大小 cue"
    },
    "high_file_size": {
        "feature": "file_size", "direction": "high", "targets": ["mel"], "top_fracs": [0.01, 0.05, 0.10, 0.20],
        "desc": "高 JPEG 文件大小 cue"
    },
    "high_bytes_per_pixel": {
        "feature": "bytes_per_pixel", "direction": "high", "targets": ["mel"], "top_fracs": [0.01, 0.05, 0.10, 0.20],
        "desc": "JPEG bytes_per_pixel cue"
    },
}


def cramer_v(table: pd.DataFrame):
    if chi2_contingency is None:
        return np.nan, np.nan
    arr = table.values
    if arr.size == 0 or arr.sum() == 0:
        return np.nan, np.nan
    chi2, p, _, _ = chi2_contingency(arr)
    n = arr.sum()
    r, k = arr.shape
    v = math.sqrt((chi2 / n) / max(1, min(k - 1, r - 1)))
    return float(v), float(p)


def enrichment_for_subset(mask, name, feature=None, threshold=None, direction=None):
    sub = df.loc[mask]
    counts = sub["dx"].value_counts().reindex(LABELS, fill_value=0)
    pct = counts / max(1, len(sub))
    rows = []
    for lab in LABELS:
        base = baseline_pct[lab]
        rows.append({
            "subset": name,
            "feature": feature,
            "threshold": threshold,
            "direction": direction,
            "n_subset": int(len(sub)),
            "label": lab,
            "count": int(counts[lab]),
            "pct": float(pct[lab]),
            "baseline_pct": float(base),
            "enrichment_ratio": float(pct[lab] / base) if base > 0 else np.nan,
            "pct_point_delta": float((pct[lab] - base) * 100),
        })
    return rows

quant_rows = []
binary_tests = []
for cname, cfg in candidate_defs.items():
    x = df[cfg["feature"]]
    for frac in cfg["top_fracs"]:
        q = frac if cfg["direction"] == "low" else 1 - frac
        threshold = float(x.quantile(q))
        mask = x <= threshold if cfg["direction"] == "low" else x >= threshold
        subset_name = f"{cname}_{'bottom' if cfg['direction']=='low' else 'top'}_{int(frac*100)}pct"
        quant_rows.extend(enrichment_for_subset(mask, subset_name, cfg["feature"], threshold, cfg["direction"]))
        table = pd.crosstab(mask.astype(int), df["dx"]).reindex(columns=LABELS, fill_value=0)
        v, p = cramer_v(table)
        binary_tests.append({
            "candidate": cname, "desc": cfg["desc"], "feature": cfg["feature"], "direction": cfg["direction"],
            "frac": frac, "threshold": threshold, "n_flagged": int(mask.sum()), "cramers_v": v, "chi2_p": p,
        })

quant_df = pd.DataFrame(quant_rows)
quant_df.to_csv(OUT_DIR / "quantile_label_enrichment.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(binary_tests).to_csv(OUT_DIR / "binary_flag_chi2_tests.csv", index=False, encoding="utf-8-sig")

# Decile crosstabs for each numeric feature.
decile_summary_rows = []
for cname, cfg in candidate_defs.items():
    feat = cfg["feature"]
    try:
        bins = pd.qcut(df[feat], 10, labels=False, duplicates="drop")
    except Exception:
        continue
    tmp = pd.crosstab(bins, df["dx"]).reindex(columns=LABELS, fill_value=0)
    tmp.to_csv(OUT_DIR / f"decile_crosstab_{cname}.csv", encoding="utf-8-sig")
    pct = tmp.div(tmp.sum(axis=1), axis=0)
    for b in tmp.index:
        row = {"candidate": cname, "feature": feat, "decile": int(b), "n": int(tmp.loc[b].sum()),
               "feature_min": float(df.loc[bins == b, feat].min()), "feature_max": float(df.loc[bins == b, feat].max())}
        for lab in LABELS:
            row[f"{lab}_pct"] = float(pct.loc[b, lab]) if tmp.loc[b].sum() else 0.0
            row[f"{lab}_count"] = int(tmp.loc[b, lab])
        decile_summary_rows.append(row)
pd.DataFrame(decile_summary_rows).to_csv(OUT_DIR / "decile_label_summary.csv", index=False, encoding="utf-8-sig")

# Class-wise numeric summaries and effect sizes.
num_features = sorted({cfg["feature"] for cfg in candidate_defs.values()} | {"edge_density", "lap_var", "dark_edge_frac", "vignette_score"})
summary_rows = []
for feat in num_features:
    groups = [g[feat].dropna().values for _, g in df.groupby("dx")]
    overall_mean = df[feat].mean()
    ss_between = sum(len(g) * (np.mean(g) - overall_mean) ** 2 for g in groups if len(g))
    ss_total = ((df[feat] - overall_mean) ** 2).sum()
    eta2 = float(ss_between / ss_total) if ss_total > 0 else np.nan
    kw_p = np.nan
    if kruskal is not None:
        try:
            kw_p = float(kruskal(*groups).pvalue)
        except Exception:
            kw_p = np.nan
    for lab, g in df.groupby("dx"):
        summary_rows.append({
            "feature": feat, "label": lab, "n": int(len(g)),
            "mean": float(g[feat].mean()), "std": float(g[feat].std()), "median": float(g[feat].median()),
            "p05": float(g[feat].quantile(0.05)), "p95": float(g[feat].quantile(0.95)),
            "eta2_label_effect": eta2, "kruskal_p": kw_p,
        })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "numeric_summary_by_label_quantitative.csv", index=False, encoding="utf-8-sig")

# Spearman correlations among artifact proxies.
corr_features = ["mean_S", "mean_V", "file_size", "bytes_per_pixel", "edge_density", "lap_var", "dark_border_frac", "dark_corner_frac", "border_center_l2", "vignette_score", "file_index"]
df[corr_features].corr(method="spearman").to_csv(OUT_DIR / "spearman_artifact_proxy_correlations.csv", encoding="utf-8-sig")

# Controlled one-vs-rest logistic regression: feature + common metadata controls.
control_rows = []
if LogisticRegression is not None:
    controls_cat = ["sex", "localization", "dx_type", "part"]
    controls_num_base = ["age"]
    # Extra artifact controls for JPEG residual checks.
    extra_controls_for = {
        "file_size": ["mean_V", "edge_density", "lap_var", "bytes_per_pixel"],
        "bytes_per_pixel": ["mean_V", "edge_density", "lap_var", "file_size"],
    }
    for cname, cfg in candidate_defs.items():
        feat = cfg["feature"]
        for lab in cfg["targets"]:
            y = (df["dx"] == lab).astype(int)
            if y.nunique() < 2:
                continue
            controls_num = controls_num_base + extra_controls_for.get(feat, [])
            num_cols = [feat] + [c for c in controls_num if c != feat]
            cat_cols = [c for c in controls_cat if c in df.columns]
            X = df[num_cols + cat_cols].copy()
            pre = ColumnTransformer([
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
                ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
            ])
            model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
            pipe = Pipeline([("pre", pre), ("model", model)])
            try:
                pipe.fit(X, y)
                coef = float(pipe.named_steps["model"].coef_[0][0])  # first numeric is candidate, per 1 SD
                pred = pipe.predict_proba(X)[:, 1]
                auc = float(roc_auc_score(y, pred))
                control_rows.append({
                    "candidate": cname, "feature": feat, "target_label": lab,
                    "coef_per_1sd_adjusted": coef, "odds_ratio_per_1sd_adjusted": float(math.exp(coef)),
                    "train_auc_descriptive": auc, "controls": "+".join(controls_num[1:] + cat_cols),
                    "n_positive": int(y.sum()), "n_total": int(len(y)),
                })
            except Exception as e:
                control_rows.append({"candidate": cname, "feature": feat, "target_label": lab, "error": str(e)})
control_df = pd.DataFrame(control_rows)
control_df.to_csv(OUT_DIR / "controlled_one_vs_rest_logistic.csv", index=False, encoding="utf-8-sig")

# Lesion-level robustness: average numeric features by lesion_id, use first dx.
lesion = df.groupby("lesion_id").agg({"dx": "first", **{feat: "mean" for feat in num_features}, "image_id": "count"}).rename(columns={"image_id": "n_images"}).reset_index()
lesion_rows = []
for cname, cfg in candidate_defs.items():
    feat = cfg["feature"]
    x = lesion[feat]
    for frac in [0.05, 0.10, 0.20]:
        q = frac if cfg["direction"] == "low" else 1 - frac
        threshold = float(x.quantile(q))
        mask = x <= threshold if cfg["direction"] == "low" else x >= threshold
        sub = lesion.loc[mask]
        counts = sub["dx"].value_counts().reindex(LABELS, fill_value=0)
        lesion_base = lesion["dx"].value_counts().reindex(LABELS, fill_value=0) / len(lesion)
        for lab in LABELS:
            pct = counts[lab] / max(1, len(sub))
            lesion_rows.append({
                "candidate": cname, "feature": feat, "direction": cfg["direction"], "frac": frac,
                "threshold": threshold, "n_lesions_flagged": int(len(sub)), "label": lab,
                "count_lesions": int(counts[lab]), "pct": float(pct), "baseline_pct": float(lesion_base[lab]),
                "enrichment_ratio": float(pct / lesion_base[lab]) if lesion_base[lab] > 0 else np.nan,
                "pct_point_delta": float((pct - lesion_base[lab]) * 100),
            })
pd.DataFrame(lesion_rows).to_csv(OUT_DIR / "lesion_level_enrichment.csv", index=False, encoding="utf-8-sig")

# file index deciles and part crosstabs.
for col, out in [("file_index", "file_index_decile_label_summary.csv")]:
    bins = pd.qcut(df[col], 10, labels=False, duplicates="drop")
    tmp = pd.crosstab(bins, df["dx"]).reindex(columns=LABELS, fill_value=0)
    pct = tmp.div(tmp.sum(axis=1), axis=0)
    rows = []
    for b in tmp.index:
        row = {"decile": int(b), "n": int(tmp.loc[b].sum()), "min_file_index": int(df.loc[bins == b, col].min()), "max_file_index": int(df.loc[bins == b, col].max())}
        for lab in LABELS:
            row[f"{lab}_count"] = int(tmp.loc[b, lab]); row[f"{lab}_pct"] = float(pct.loc[b, lab])
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / out, index=False, encoding="utf-8-sig")

part_tab = pd.crosstab(df["part"], df["dx"]).reindex(columns=LABELS, fill_value=0)
part_tab.to_csv(OUT_DIR / "part_label_crosstab.csv", encoding="utf-8-sig")
part_pct = part_tab.div(part_tab.sum(axis=1), axis=0)
part_pct.to_csv(OUT_DIR / "part_label_pct.csv", encoding="utf-8-sig")

# Decision heuristics based on top/bottom 5% enrichment + adjusted OR when available.
decisions = []
q5 = quant_df[quant_df["subset"].str.contains("_5pct")].copy()
ctrl = control_df.set_index(["candidate", "target_label"]) if not control_df.empty and "candidate" in control_df else pd.DataFrame()
for cname, cfg in candidate_defs.items():
    rows = q5[(q5["subset"].str.startswith(cname + "_")) & (q5["label"].isin(cfg["targets"]))]
    max_delta = float(rows["pct_point_delta"].max()) if len(rows) else np.nan
    max_er = float(rows["enrichment_ratio"].max()) if len(rows) else np.nan
    target_metrics = []
    adjusted_support = []
    for _, r in rows.iterrows():
        item = {"label": r["label"], "pct": r["pct"], "baseline_pct": r["baseline_pct"], "delta_pp": r["pct_point_delta"], "enrichment_ratio": r["enrichment_ratio"], "n_subset": r["n_subset"]}
        if isinstance(ctrl, pd.DataFrame) and not ctrl.empty and (cname, r["label"]) in ctrl.index:
            c = ctrl.loc[(cname, r["label"])]
            if isinstance(c, pd.DataFrame): c = c.iloc[0]
            item["adjusted_or_per_1sd"] = c.get("odds_ratio_per_1sd_adjusted", np.nan)
            adjusted_support.append(item["adjusted_or_per_1sd"])
        target_metrics.append(item)
    # heuristic judgement
    if max_delta >= 15 or max_er >= 1.6:
        judgement = "成立"
    elif max_delta >= 7 or max_er >= 1.25:
        judgement = "待确认"
    else:
        judgement = "不成立/弱"
    decisions.append({"candidate": cname, "desc": cfg["desc"], "feature": cfg["feature"], "direction": cfg["direction"], "targets": cfg["targets"], "target_metrics_top_or_bottom_5pct": target_metrics, "max_delta_pp": max_delta, "max_enrichment_ratio": max_er, "judgement": judgement})

with open(OUT_DIR / "ham10000_quantitative_shortcut_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "dataset": "HAM10000", "dataset_dir": str(DATASET_DIR), "feature_source": str(FEATURE_PATH), "metadata_source": str(META_PATH),
        "n_images": int(N), "n_lesions": int(df["lesion_id"].nunique()), "baseline_counts": baseline.to_dict(),
        "baseline_pct": {k: float(v) for k, v in baseline_pct.to_dict().items()}, "decisions": decisions,
        "outputs": [p.name for p in OUT_DIR.iterdir()]
    }, f, ensure_ascii=False, indent=2)

# Markdown report.
def pct(x): return f"{100*x:.1f}%"
def fmt(x):
    if pd.isna(x): return "NA"
    return f"{x:.3g}"

lines = []
lines.append("# HAM10000 数据集特有候选捷径定量分析报告")
lines.append("")
lines.append(f"- 数据集名称：HAM10000")
lines.append(f"- 数据集地址：`{DATASET_DIR}`")
lines.append(f"- 使用特征来源：`{FEATURE_PATH}`")
lines.append(f"- 使用元数据来源：`{META_PATH}`（仅作控制变量，不重新分析常见元数据捷径）")
lines.append(f"- 样本数：{N:,} 张图像；lesion_id 数：{df['lesion_id'].nunique():,}")
lines.append(f"- 基线标签比例：" + ", ".join([f"{lab}={pct(baseline_pct[lab])}" for lab in LABELS]))
lines.append("")
for d in decisions:
    lines.append(f"## {d['desc']}（{d['candidate']}）")
    lines.append(f"- 特征：`{d['feature']}`；方向：{d['direction']}；强度判断：**{d['judgement']}**")
    rows = q5[(q5["subset"].str.startswith(d["candidate"] + "_")) & (q5["label"].isin(d["targets"]))]
    if len(rows):
        for _, r in rows.iterrows():
            lines.append(f"- 5% 极端分位：{r['label']} = {pct(r['pct'])}，基线 {pct(r['baseline_pct'])}，delta={r['pct_point_delta']:.1f}pp，ER={r['enrichment_ratio']:.2f}，n={int(r['n_subset'])}")
    bt = pd.read_csv(OUT_DIR / "binary_flag_chi2_tests.csv")
    bt5 = bt[(bt["candidate"] == d["candidate"]) & (abs(bt["frac"] - 0.05) < 1e-9)]
    if len(bt5):
        b = bt5.iloc[0]
        lines.append(f"- 二值极端分位 vs 标签：Cramer's V={fmt(b['cramers_v'])}, chi2 p={fmt(b['chi2_p'])}")
    if not control_df.empty:
        subc = control_df[control_df["candidate"] == d["candidate"]]
        if len(subc):
            txt = []
            for _, c in subc.iterrows():
                if 'odds_ratio_per_1sd_adjusted' in c and pd.notna(c.get('odds_ratio_per_1sd_adjusted', np.nan)):
                    txt.append(f"{c['target_label']}: adjusted OR/1SD={c['odds_ratio_per_1sd_adjusted']:.2f}")
            if txt:
                lines.append("- 控制 age/sex/localization/dx_type/part 后描述性 one-vs-rest：" + "; ".join(txt))
    lines.append("")
lines.append("## 文件名序号/采集批次 cue")
fi = pd.read_csv(OUT_DIR / "file_index_decile_label_summary.csv")
for lab in ["mel", "nv", "bkl", "bcc"]:
    cols = [f"{lab}_pct"]
    if cols[0] in fi.columns:
        rowmax = fi.loc[fi[cols[0]].idxmax()]
        rowmin = fi.loc[fi[cols[0]].idxmin()]
        lines.append(f"- {lab}: decile max={pct(rowmax[cols[0]])} (D{int(rowmax['decile'])}, {int(rowmax['min_file_index'])}-{int(rowmax['max_file_index'])}); min={pct(rowmin[cols[0]])} (D{int(rowmin['decile'])})")
lines.append("- 解释：该项作为采集批次/排序与 split 泄漏风险提示，不作为疾病语义捷径。")
lines.append("")
lines.append("## 图像 part 文件夹 cue")
lines.append("- 仅做 split 泄漏风险备注；详见 `part_label_crosstab.csv` 与 `part_label_pct.csv`。")
lines.append("")
lines.append("## 限制")
lines.append("- 本轮只基于 Identify Agent 已产出的低成本全量图像特征与 HAM10000 元数据控制变量；未重新进行病灶分割、模型训练或反事实图像干预。")
lines.append("- 颜色/亮度类 cue 可能混合真实病灶颜色、曝光、采集设备与背景/边框 artifact；控制元数据后仍只能作为描述性关联，不等同因果。")
lines.append("- lesion_id 稳健性使用 lesion-level 聚合检查，不能完全替代严格的按 lesion 分组建模或外部验证集。")
lines.append("")
lines.append("## 输出文件")
for p in sorted(OUT_DIR.iterdir()):
    lines.append(f"- `{p}`")

(OUT_DIR / "HAM10000_quantitative_shortcut_report.md").write_text("\n".join(lines), encoding="utf-8")

print(f"Wrote outputs to: {OUT_DIR}")
print((OUT_DIR / "HAM10000_quantitative_shortcut_report.md").read_text(encoding="utf-8")[:5000])
