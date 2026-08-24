import os, json, math, csv, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import cv2
except Exception:
    cv2 = None
try:
    from scipy.stats import kruskal, chi2_contingency
except Exception:
    kruskal = chi2_contingency = None
try:
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
except Exception:
    local_binary_pattern = graycomatrix = graycoprops = None
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
except Exception:
    LogisticRegression = None

BASE = Path(r"E:\小龙虾")
ROOT = BASE / "Plant Disease Recognition Dataset"
PHASE1 = BASE / "plant_phase1_shortcut_summary.json"
OUT = BASE / "plant_disease_quantitative_shortcut_results"
OUT.mkdir(parents=True, exist_ok=True)
FEATURE_CSV = OUT / "plant_candidate_features.csv"

LABELS = ["Healthy", "Powdery", "Rust"]
SPLITS = ["Train", "Validation", "Test"]
IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.webp'}


def list_images():
    rows=[]
    for split in SPLITS:
        sd = ROOT/split
        if not sd.exists():
            continue
        # Dataset has an extra nested split folder, e.g. Train/Train/Healthy.
        candidates = []
        nested = sd / split
        if nested.exists() and nested.is_dir():
            candidates.append(nested)
        candidates.append(sd)
        seen=set()
        for base_dir in candidates:
            for lab in LABELS:
                labdir = base_dir / lab
                if not labdir.exists() or not labdir.is_dir():
                    continue
                for p in labdir.rglob('*'):
                    if p.suffix.lower() in IMG_EXTS and str(p) not in seen:
                        seen.add(str(p))
                        rows.append({'path': str(p), 'filename': p.name, 'split': split, 'label': lab})
    return rows


def shannon_entropy(values, bins=256, value_range=(0,255)):
    hist,_ = np.histogram(values, bins=bins, range=value_range)
    prob = hist.astype(float) / max(1, hist.sum())
    prob = prob[prob>0]
    return float(-(prob*np.log2(prob)).sum())


def rgb_to_hsv_np(rgb):
    if cv2 is not None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        return hsv.astype(np.float32)
    # fallback approximate
    arr = rgb.astype(np.float32)/255.0
    mx=arr.max(axis=2); mn=arr.min(axis=2); diff=mx-mn
    h=np.zeros_like(mx)
    mask=diff!=0
    r,g,b=arr[:,:,0],arr[:,:,1],arr[:,:,2]
    h[(mx==r)&mask]=(60*((g[(mx==r)&mask]-b[(mx==r)&mask])/diff[(mx==r)&mask])+360)%360
    h[(mx==g)&mask]=(60*((b[(mx==g)&mask]-r[(mx==g)&mask])/diff[(mx==g)&mask])+120)%360
    h[(mx==b)&mask]=(60*((r[(mx==b)&mask]-g[(mx==b)&mask])/diff[(mx==b)&mask])+240)%360
    s=np.where(mx==0,0,diff/mx)
    v=mx
    return np.stack([h/2, s*255, v*255], axis=2).astype(np.float32)


def compute_lbp(gray_small):
    if local_binary_pattern is not None:
        lbp = local_binary_pattern(gray_small, P=8, R=1, method='uniform')
        maxv = int(lbp.max()) + 1
        hist,_ = np.histogram(lbp, bins=maxv, range=(0,maxv))
        p = hist / max(1,hist.sum())
        p = p[p>0]
        return float(-(p*np.log2(p)).sum())
    # simple local complexity fallback: entropy of 3x3 sign patterns
    g = gray_small.astype(np.int16)
    center = g[1:-1,1:-1]
    code = np.zeros_like(center, dtype=np.uint8)
    offsets=[(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    for i,(dy,dx) in enumerate(offsets):
        code |= ((g[1+dy:g.shape[0]-1+dy,1+dx:g.shape[1]-1+dx] >= center).astype(np.uint8) << i)
    return shannon_entropy(code, bins=256, value_range=(0,255))


def patch_lbp_stats(gray_small, grid=4):
    h,w = gray_small.shape
    vals=[]
    for gy in range(grid):
        for gx in range(grid):
            y0,y1 = int(gy*h/grid), int((gy+1)*h/grid)
            x0,x1 = int(gx*w/grid), int((gx+1)*w/grid)
            patch=gray_small[y0:y1,x0:x1]
            if patch.size>64:
                vals.append(compute_lbp(patch))
    vals=np.array(vals,float)
    return float(vals.mean()), float(vals.std()), float(vals.min()), float(vals.max())


def glcm_features(gray_small):
    q = np.clip((gray_small.astype(np.float32)/32).astype(np.uint8),0,7)
    if graycomatrix is not None:
        glcm = graycomatrix(q, distances=[1,2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=8, symmetric=True, normed=True)
        out={}
        for prop in ['contrast','homogeneity','energy','correlation']:
            out['glcm_'+prop] = float(graycoprops(glcm, prop).mean())
        return out
    # Built-in lightweight fallback: 8-level GLCM, average over 4 neighbor offsets.
    offsets = [(0,1),(1,0),(1,1),(1,-1)]
    mats=[]
    h,w=q.shape
    for dy,dx in offsets:
        if dx >= 0:
            a = q[max(0,dy):h, 0:w-dx]
            b = q[0:h-dy, dx:w]
        else:
            a = q[max(0,dy):h, -dx:w]
            b = q[0:h-dy, 0:w+dx]
        if a.size == 0 or b.size == 0:
            continue
        mat=np.zeros((8,8), dtype=np.float64)
        np.add.at(mat, (a.ravel(), b.ravel()), 1)
        mat = mat + mat.T
        mat = mat / max(1.0, mat.sum())
        mats.append(mat)
    if not mats:
        return {"glcm_contrast": np.nan, "glcm_homogeneity": np.nan, "glcm_energy": np.nan, "glcm_correlation": np.nan}
    P=np.mean(mats, axis=0)
    i,j=np.indices(P.shape)
    contrast=float(((i-j)**2 * P).sum())
    homogeneity=float((P/(1.0+np.abs(i-j))).sum())
    energy=float(np.sqrt((P**2).sum()))
    mi=float((i*P).sum()); mj=float((j*P).sum())
    si=float(np.sqrt((((i-mi)**2)*P).sum())); sj=float(np.sqrt((((j-mj)**2)*P).sum()))
    correlation=float((((i-mi)*(j-mj)*P).sum())/(si*sj+1e-9))
    return {"glcm_contrast": contrast, "glcm_homogeneity": homogeneity, "glcm_energy": energy, "glcm_correlation": correlation}


def frequency_features(gray_small):
    arr = gray_small.astype(np.float32)
    arr = arr - arr.mean()
    f = np.fft.fftshift(np.fft.fft2(arr))
    power = np.abs(f)**2
    h,w = power.shape
    yy,xx=np.ogrid[:h,:w]
    cy,cx=h/2,w/2
    rr=np.sqrt((yy-cy)**2+(xx-cx)**2)
    maxr=np.sqrt(cy**2+cx**2)
    total=power.sum()+1e-9
    return {
        'fft_high_freq_ratio': float(power[rr > 0.35*maxr].sum()/total),
        'fft_mid_freq_ratio': float(power[(rr > 0.15*maxr)&(rr <= 0.35*maxr)].sum()/total),
        'fft_low_freq_ratio': float(power[rr <= 0.15*maxr].sum()/total),
    }


def feature_one(row):
    p=Path(row['path'])
    img=Image.open(p).convert('RGB')
    w,h=img.size
    # resize to fixed max side for stable low-cost features
    max_side=512
    scale=min(1.0, max_side/max(w,h))
    if scale<1:
        img_small=img.resize((int(w*scale), int(h*scale)), Image.Resampling.BILINEAR)
    else:
        img_small=img.copy()
    rgb=np.asarray(img_small).astype(np.uint8)
    H,W=rgb.shape[:2]
    gray=np.asarray(img_small.convert('L')).astype(np.uint8)
    hsv=rgb_to_hsv_np(rgb)
    r,g,b=rgb[:,:,0].astype(float),rgb[:,:,1].astype(float),rgb[:,:,2].astype(float)
    luma=0.299*r+0.587*g+0.114*b
    # color masks: approximate pathological/color block proxies, aligned with phase1 names
    green_mask=(g>r*1.05)&(g>b*1.05)&(g>60)
    yellow_mask=(r>110)&(g>90)&(b<130)&(np.abs(r-g)<70)
    brown_mask=(r>70)&(g>35)&(b<95)&(r>g*1.05)&(g>b*1.05)&((r-g)<95)
    white_mask=(r>170)&(g>170)&(b>170)&((np.maximum.reduce([r,g,b])-np.minimum.reduce([r,g,b]))<45)
    dark_mask=luma<45
    # foreground: non-bright/near-white background + green/brown/yellow/dark object proxy
    fg_mask = (green_mask | yellow_mask | brown_mask | dark_mask | ((hsv[:,:,1]>35)&(hsv[:,:,2]<245)))
    fg_area=float(fg_mask.mean())
    if fg_mask.any():
        ys,xs=np.where(fg_mask)
        fg_cx=float(xs.mean()/max(1,W-1)); fg_cy=float(ys.mean()/max(1,H-1))
        bbox_area=float(((xs.max()-xs.min()+1)*(ys.max()-ys.min()+1))/(W*H))
        touch=float((xs.min()==0) or (ys.min()==0) or (xs.max()==W-1) or (ys.max()==H-1))
    else:
        fg_cx=fg_cy=bbox_area=touch=np.nan
    # edge/sharpness
    if cv2 is not None:
        lap_var=float(cv2.Laplacian(gray, cv2.CV_64F).var())
        med=np.median(gray)
        edges=cv2.Canny(gray, int(max(0,0.66*med)), int(min(255,1.33*med)))
        edge_density=float((edges>0).mean())
    else:
        gy,gx=np.gradient(gray.astype(float)); mag=np.sqrt(gx*gx+gy*gy)
        lap_var=float(np.var(np.gradient(gx)[1]+np.gradient(gy)[0]))
        edge_density=float((mag>np.percentile(mag,90)).mean())
    # borders
    bw=max(4,int(min(H,W)*0.05))
    border=np.concatenate([rgb[:bw,:,:].reshape(-1,3), rgb[-bw:,:,:].reshape(-1,3), rgb[:, :bw, :].reshape(-1,3), rgb[:, -bw:, :].reshape(-1,3)], axis=0).astype(float)
    out={**row}
    out.update({
        'w':w,'h':h,'aspect':w/h,'size_bytes':p.stat().st_size,
        'mean_r':float(r.mean()),'mean_g':float(g.mean()),'mean_b':float(b.mean()),
        'std_r':float(r.std()),'std_g':float(g.std()),'std_b':float(b.std()),
        'mean_h':float(hsv[:,:,0].mean()),'mean_s':float(hsv[:,:,1].mean()),'mean_v':float(hsv[:,:,2].mean()),
        'std_luma':float(luma.std()),'entropy':shannon_entropy(gray),
        'lap_var':lap_var,'edge_density':edge_density,
        'lbp_entropy':compute_lbp(gray),
        'green_ratio':float(green_mask.mean()),'yellow_ratio':float(yellow_mask.mean()),
        'brown_ratio':float(brown_mask.mean()),'white_ratio':float(white_mask.mean()),'dark_ratio':float(dark_mask.mean()),
        'fg_area':fg_area,'fg_cx':fg_cx,'fg_cy':fg_cy,'fg_bbox_area':bbox_area,'fg_border_touch':touch,
        'border_r':float(border[:,0].mean()),'border_g':float(border[:,1].mean()),'border_b':float(border[:,2].mean()),'border_std':float(border.std()),
    })
    plm,pls,plmin,plmax=patch_lbp_stats(gray,4)
    out.update({'patch_lbp_entropy_mean':plm,'patch_lbp_entropy_std':pls,'patch_lbp_entropy_min':plmin,'patch_lbp_entropy_max':plmax})
    out.update(glcm_features(gray))
    out.update(frequency_features(gray))
    # derived proxies
    out['blue_excess']=float(out['mean_b'] - (out['mean_r']+out['mean_g'])/2)
    out['rg_temp_proxy']=float(out['mean_r'] - out['mean_b'])
    out['border_blue_excess']=float(out['border_b'] - (out['border_r']+out['border_g'])/2)
    return out


def eta2_anova(df, feat):
    overall=df[feat].mean()
    ss_between=sum(len(g)*(g[feat].mean()-overall)**2 for _,g in df.groupby('label'))
    ss_total=((df[feat]-overall)**2).sum()
    return float(ss_between/ss_total) if ss_total>0 else np.nan

def cohend(a,b):
    a=np.asarray(a.dropna(),float); b=np.asarray(b.dropna(),float)
    if len(a)<2 or len(b)<2: return np.nan
    sp=math.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return float((a.mean()-b.mean())/sp) if sp>0 else np.nan

def cramer(table):
    if chi2_contingency is None: return np.nan,np.nan
    chi2,p,_,_=chi2_contingency(table.values)
    n=table.values.sum(); r,k=table.shape
    return float(math.sqrt((chi2/n)/max(1,min(r-1,k-1)))), float(p)

def q_enrich(df, feat, direction, fracs=(0.05,0.1,0.2)):
    rows=[]
    base=df['label'].value_counts(normalize=True).reindex(LABELS,fill_value=0)
    for frac in fracs:
        thr=float(df[feat].quantile(frac if direction=='low' else 1-frac))
        mask=df[feat]<=thr if direction=='low' else df[feat]>=thr
        sub=df[mask]
        pct=sub['label'].value_counts(normalize=True).reindex(LABELS,fill_value=0)
        cnt=sub['label'].value_counts().reindex(LABELS,fill_value=0)
        tab=pd.crosstab(mask.astype(int), df['label']).reindex(columns=LABELS,fill_value=0)
        cv,p=cramer(tab)
        for lab in LABELS:
            rows.append({'feature':feat,'direction':direction,'frac':frac,'threshold':thr,'n':int(mask.sum()),'label':lab,
                         'count':int(cnt[lab]),'pct':float(pct[lab]),'baseline_pct':float(base[lab]),
                         'delta_pp':float((pct[lab]-base[lab])*100),'enrichment_ratio':float(pct[lab]/base[lab]) if base[lab]>0 else np.nan,
                         'cramers_v':cv,'chi2_p':p})
    return rows


def simple_cv_classify(df, feature_sets):
    rows=[]
    if LogisticRegression is None: return pd.DataFrame(rows)
    y=df['label'].map({lab:i for i,lab in enumerate(LABELS)}).values
    for name, feats in feature_sets.items():
        X=df[feats].replace([np.inf,-np.inf],np.nan).fillna(df[feats].median()).values
        if len(df)<10: continue
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model=Pipeline([('sc',StandardScaler()),('lr',LogisticRegression(max_iter=1000,class_weight='balanced'))])
        try:
            pred=cross_val_predict(model,X,y,cv=cv)
            rows.append({'feature_set':name,'features':'+'.join(feats),'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred))})
        except Exception as e:
            rows.append({'feature_set':name,'features':'+'.join(feats),'error':str(e)})
    return pd.DataFrame(rows)


def main():
    if FEATURE_CSV.exists():
        df=pd.read_csv(FEATURE_CSV)
    else:
        imgs=list_images()
        print('images',len(imgs))
        rows=[]
        for i,row in enumerate(imgs,1):
            if i%50==0: print('processing',i,'/',len(imgs), flush=True)
            try:
                rows.append(feature_one(row))
            except Exception as e:
                print('ERROR', row['path'], e)
        df=pd.DataFrame(rows)
        df.to_csv(FEATURE_CSV,index=False,encoding='utf-8-sig')
    # summaries
    summary=[]
    candidate_features=['lbp_entropy','patch_lbp_entropy_mean','patch_lbp_entropy_std','glcm_contrast','glcm_homogeneity','glcm_energy','fft_high_freq_ratio','lap_var','edge_density','mean_b','mean_r','mean_g','mean_h','mean_s','mean_v','blue_excess','rg_temp_proxy','brown_ratio','green_ratio','yellow_ratio','white_ratio','fg_area','fg_cx','fg_cy','fg_bbox_area','border_r','border_g','border_b','border_std','aspect','size_bytes']
    for feat in candidate_features:
        if feat not in df: continue
        groups=[g[feat].dropna().values for _,g in df.groupby('label')]
        pval=np.nan
        if kruskal is not None:
            try: pval=float(kruskal(*groups).pvalue)
            except Exception: pass
        e=eta2_anova(df,feat)
        for lab,g in df.groupby('label'):
            summary.append({'feature':feat,'label':lab,'n':len(g),'mean':float(g[feat].mean()),'std':float(g[feat].std()),'median':float(g[feat].median()),'p05':float(g[feat].quantile(.05)),'p95':float(g[feat].quantile(.95)),'eta2':e,'kruskal_p':pval})
    sumdf=pd.DataFrame(summary)
    sumdf.to_csv(OUT/'numeric_summary_by_label.csv',index=False,encoding='utf-8-sig')
    # pairwise d
    ds=[]
    for feat in candidate_features:
        if feat not in df: continue
        for a in LABELS:
            for b in LABELS:
                if a>=b: continue
                ds.append({'feature':feat,'pair':f'{a}-{b}','cohen_d_a_minus_b':cohend(df[df.label==a][feat],df[df.label==b][feat])})
    pd.DataFrame(ds).to_csv(OUT/'pairwise_cohen_d.csv',index=False,encoding='utf-8-sig')
    # quantile enrichment focused on requested directions
    qdefs=[
        ('lbp_entropy','low'),('patch_lbp_entropy_mean','low'),('glcm_homogeneity','high'),('glcm_contrast','low'),('fft_high_freq_ratio','low'),
        ('lap_var','low'),('edge_density','low'),('edge_density','high'),('mean_b','high'),('blue_excess','high'),('mean_s','low'),('mean_v','high'),
        ('brown_ratio','high'),('green_ratio','high'),('green_ratio','low'),('yellow_ratio','high'),('white_ratio','high'),
        ('fg_area','high'),('fg_area','low'),('border_b','high'),('border_std','low'),('border_r','high'),('aspect','high'),('aspect','low')]
    qrows=[]
    for feat,dirn in qdefs:
        if feat in df: qrows += q_enrich(df,feat,dirn)
    qdf=pd.DataFrame(qrows)
    qdf.to_csv(OUT/'quantile_label_enrichment.csv',index=False,encoding='utf-8-sig')
    # split robustness: train/val/test means
    split_summary=df.groupby(['split','label'])[candidate_features].mean(numeric_only=True).reset_index()
    split_summary.to_csv(OUT/'split_label_feature_means.csv',index=False,encoding='utf-8-sig')
    # simple descriptive classifiers
    clf=simple_cv_classify(df,{
        'texture_only':['lbp_entropy','patch_lbp_entropy_mean','patch_lbp_entropy_std','glcm_contrast','glcm_homogeneity','fft_high_freq_ratio'],
        'sharp_edge_only':['lap_var','edge_density','fft_high_freq_ratio'],
        'color_only':['mean_r','mean_g','mean_b','mean_h','mean_s','mean_v','blue_excess','rg_temp_proxy'],
        'spot_color_only':['brown_ratio','green_ratio','yellow_ratio','white_ratio','dark_ratio'],
        'crop_fg_only':['fg_area','fg_cx','fg_cy','fg_bbox_area','aspect'],
        'border_aspect_only':['border_r','border_g','border_b','border_std','aspect'],
    })
    clf.to_csv(OUT/'descriptive_feature_set_cv_classification.csv',index=False,encoding='utf-8-sig')
    # correlations
    corr_feats=[f for f in candidate_features if f in df]
    df[corr_feats].corr(method='spearman').to_csv(OUT/'spearman_candidate_feature_correlations.csv',encoding='utf-8-sig')
    # build decisions manually based on stats
    def m(feat, lab):
        r=sumdf[(sumdf.feature==feat)&(sumdf.label==lab)].iloc[0]
        return r
    def topq(feat,dirn,lab,frac=.05):
        r=qdf[(qdf.feature==feat)&(qdf.direction==dirn)&(qdf.frac==frac)&(qdf.label==lab)]
        return None if r.empty else r.iloc[0]
    decisions=[]
    # candidate group helper
    groups = {
        'texture_smoothness_local_complexity': {'features':['lbp_entropy','patch_lbp_entropy_mean','glcm_contrast','glcm_homogeneity','fft_high_freq_ratio'], 'judgement':'成立', 'pathology_vs_artifact':'更可能与 Powdery 病害粉状覆盖/局部纹理变平滑有关，但也可被成像模糊/压缩风格放大；需用灰度纹理-only 与模糊化对照验证。'},
        'sharpness_edge_density': {'features':['lap_var','edge_density','fft_high_freq_ratio'], 'judgement':'成立/待确认', 'pathology_vs_artifact':'Powdery 更低锐度/边缘密度，Rust 更高；可能同时包含病害纹理差异与拍摄清晰度/焦距 artifact。'},
        'color_channel_color_temperature': {'features':['mean_b','blue_excess','mean_r','mean_g','mean_h','mean_s','mean_v'], 'judgement':'成立/待确认', 'pathology_vs_artifact':'Powdery 蓝通道偏高可能是白粉病症状的颜色表现，也可能是白平衡/色温采集风格；需灰度化与颜色归一化复测。'},
        'lesion_color_patch_ratio': {'features':['brown_ratio','green_ratio','yellow_ratio','white_ratio'], 'judgement':'成立', 'pathology_vs_artifact':'Rust brown_ratio 明显更高，最像真实病斑症状；但模型仍可能把局部棕色色块当作 shortcut，需 hue-shift/病斑遮挡验证。'},
        'foreground_crop_area': {'features':['fg_area','aspect','fg_cx','fg_cy'], 'judgement':'成立/待确认', 'pathology_vs_artifact':'Healthy 前景覆盖更高，可能来自叶片健康样本构图/裁剪尺度，不是病理本身；属于裁剪/采集风格风险。'},
        'border_background_aspect': {'features':['border_r','border_b','border_std','aspect'], 'judgement':'待确认/辅助', 'pathology_vs_artifact':'边框背景和宽高比为弱到中等关联，偏非病理采集/构图 artifact。'},
        'filename_exif_duplicates_leakage': {'features':[], 'judgement':'不成立/未发现', 'pathology_vs_artifact':'EXIF 基本缺失、文件名无显式类别 token、未发现精确重复或 dHash<=5 跨 split/跨标签近重复；仅作为限制说明。'},
    }
    for name,info in groups.items():
        feats=[]
        for feat in info['features']:
            if feat in sumdf.feature.unique():
                eta=float(sumdf[sumdf.feature==feat]['eta2'].iloc[0]); p=float(sumdf[sumdf.feature==feat]['kruskal_p'].iloc[0])
                means={lab:float(m(feat,lab)['mean']) for lab in LABELS}
                feats.append({'feature':feat,'eta2':eta,'kruskal_p':p,'means':means})
        decisions.append({'candidate':name, **info, 'feature_metrics':feats})
    with open(OUT/'plant_quantitative_shortcut_summary.json','w',encoding='utf-8') as f:
        json.dump({'dataset':'Plant Disease Recognition Dataset','dataset_dir':str(ROOT),'n':len(df),'counts':df.groupby(['split','label']).size().unstack(fill_value=0).to_dict(), 'decisions':decisions,'outputs':[p.name for p in OUT.iterdir()]},f,ensure_ascii=False,indent=2)
    # report
    def pct(x): return f'{100*x:.1f}%'
    lines=[]
    lines.append('# Plant Disease Recognition Dataset 定量分析报告')
    lines.append('')
    lines.append(f'- 数据集名称：Plant Disease Recognition Dataset')
    lines.append(f'- 数据集地址：`{ROOT}`')
    lines.append(f'- 样本数：{len(df)}；类别计数：' + ', '.join([f"{lab}={(df.label==lab).sum()}" for lab in LABELS]))
    lines.append(f'- 使用数据/特征来源：一阶段摘要 `{PHASE1}` + 针对已给候选 cue 重新计算的逐图特征 `{FEATURE_CSV}`')
    lines.append('- 本轮未扩展新候选，不重做文件名/EXIF/重复识别；只对指定 cue 做强度定量、分位富集、split 稳定性和验证方案。')
    lines.append('')
    for d in decisions:
        lines.append(f"## {d['candidate']}")
        lines.append(f"- 强度判断：**{d['judgement']}**")
        lines.append(f"- 病理 vs artifact：{d['pathology_vs_artifact']}")
        for fm in d['feature_metrics']:
            ms=', '.join([f"{lab}={fm['means'][lab]:.4g}" for lab in LABELS])
            lines.append(f"- `{fm['feature']}`: eta2={fm['eta2']:.3f}, p={fm['kruskal_p']:.2e}, means: {ms}")
        # targeted q examples
        if d['candidate']=='texture_smoothness_local_complexity':
            for feat,dirn,lab in [('lbp_entropy','low','Powdery'),('patch_lbp_entropy_mean','low','Powdery'),('glcm_homogeneity','high','Powdery')]:
                r=topq(feat,dirn,lab)
                if r is not None: lines.append(f"- 5% 分位 `{feat}` {dirn}: {lab}={pct(r.pct)}，基线={pct(r.baseline_pct)}，delta={r.delta_pp:.1f}pp，ER={r.enrichment_ratio:.2f}, n={int(r.n)}")
            lines.append('- 建议验证：16×16/32×32 patch-level LBP/GLCM 热图；随机遮挡低熵区域后评估 Powdery 置信度下降；灰度纹理-only 分类与颜色归一化后复测。')
        elif d['candidate']=='sharpness_edge_density':
            for feat,dirn,lab in [('lap_var','low','Powdery'),('edge_density','low','Powdery'),('edge_density','high','Rust')]:
                r=topq(feat,dirn,lab)
                if r is not None: lines.append(f"- 5% 分位 `{feat}` {dirn}: {lab}={pct(r.pct)}，基线={pct(r.baseline_pct)}，delta={r.delta_pp:.1f}pp，ER={r.enrichment_ratio:.2f}, n={int(r.n)}")
            lines.append('- 建议验证：统一 Gaussian blur/锐化后重测；只输入 Canny/边缘图训练轻量分类器；比较原图 vs 灰度边缘图。')
        elif d['candidate']=='color_channel_color_temperature':
            for feat,dirn,lab in [('mean_b','high','Powdery'),('blue_excess','high','Powdery')]:
                r=topq(feat,dirn,lab)
                if r is not None: lines.append(f"- 5% 分位 `{feat}` {dirn}: {lab}={pct(r.pct)}，基线={pct(r.baseline_pct)}，delta={r.delta_pp:.1f}pp，ER={r.enrichment_ratio:.2f}, n={int(r.n)}")
            lines.append('- 建议验证：灰度化、per-image color constancy/白平衡归一化、叶片前景内 HSV 复测；若性能大幅下降则存在颜色捷径风险。')
        elif d['candidate']=='lesion_color_patch_ratio':
            for feat,dirn,lab in [('brown_ratio','high','Rust'),('green_ratio','high','Healthy'),('white_ratio','high','Powdery')]:
                r=topq(feat,dirn,lab)
                if r is not None: lines.append(f"- 5% 分位 `{feat}` {dirn}: {lab}={pct(r.pct)}，基线={pct(r.baseline_pct)}，delta={r.delta_pp:.1f}pp，ER={r.enrichment_ratio:.2f}, n={int(r.n)}")
            lines.append('- 建议验证：局部棕/白病斑 mask 遮挡、hue-shift、只保留病斑色块/去病斑对照；区分真实病害症状与颜色块 shortcut。')
        elif d['candidate']=='foreground_crop_area':
            for feat,dirn,lab in [('fg_area','high','Healthy'),('fg_area','low','Powdery'),('aspect','high','Powdery')]:
                r=topq(feat,dirn,lab)
                if r is not None: lines.append(f"- 5% 分位 `{feat}` {dirn}: {lab}={pct(r.pct)}，基线={pct(r.baseline_pct)}，delta={r.delta_pp:.1f}pp，ER={r.enrichment_ratio:.2f}, n={int(r.n)}")
            lines.append('- 建议验证：中心裁剪/随机缩放/统一叶片 bbox resize；遮挡背景边缘，观察 Healthy 置信度是否依赖大面积前景。')
        elif d['candidate']=='border_background_aspect':
            for feat,dirn,lab in [('border_std','low','Powdery'),('border_b','high','Powdery'),('aspect','high','Powdery')]:
                r=topq(feat,dirn,lab)
                if r is not None: lines.append(f"- 5% 分位 `{feat}` {dirn}: {lab}={pct(r.pct)}，基线={pct(r.baseline_pct)}，delta={r.delta_pp:.1f}pp，ER={r.enrichment_ratio:.2f}, n={int(r.n)}")
        lines.append('')
    if not clf.empty:
        lines.append('## 描述性 feature-set CV 分类实验（非最终模型，仅衡量 cue 可分性）')
        for _,r in clf.iterrows():
            if 'balanced_accuracy' in r and pd.notna(r.get('balanced_accuracy',np.nan)):
                lines.append(f"- {r['feature_set']}: acc={r['accuracy']:.3f}, balanced_acc={r['balanced_accuracy']:.3f}")
        lines.append('')
    lines.append('## 泄漏与限制说明')
    lines.append('- 文件名/EXIF/近重复：沿用一阶段结论；EXIF 基本缺失，文件名近似随机十六进制，无显式类别 token，精确重复 0，dHash<=5 跨 split/跨标签近重复 0。')
    lines.append('- 本轮重新计算的是已给候选 cue 的逐图/patch-level 指标，用于分位数和效应量定量；不是重新做候选搜索。')
    lines.append('- 病理相关信号与 shortcut 不互斥：真实病斑颜色/纹理也可能被模型以非稳健的颜色块、平滑度或裁剪风格方式利用。')
    lines.append('- 数据量较小，Validation/Test 各类仅 20/50，定量结论应优先作为风险排序，后续需反事实干预验证。')
    lines.append('')
    lines.append('## 输出文件')
    for p in sorted(OUT.iterdir()): lines.append(f'- `{p}`')
    (OUT/'Plant_Disease_Recognition_quantitative_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote', OUT)
    print((OUT/'Plant_Disease_Recognition_quantitative_report.md').read_text(encoding='utf-8')[:6000])

if __name__=='__main__':
    main()
