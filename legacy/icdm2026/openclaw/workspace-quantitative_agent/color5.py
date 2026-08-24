##最终版本
import os
import sys
import json
import base64
import random
import re
import cv2
import io
import warnings
import hashlib
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamModel,
    SamProcessor,
    VitMatteForImageMatting,
    VitMatteImageProcessor
)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==========================================
# 1. 全局配置与模型路径
# ==========================================
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
VLM_MODEL = "qwen-vl-plus"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "Color_Audit_Results3"

# --- 挖掘 / 构造 / 训练参数 ---
AUDIT_SAMPLE_COUNT_TRAIN = 300
OOD_TARGET_COUNT = 200
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

SHORTCUT_THRESHOLD = 0.5

# --- OOD 视觉大模型本地路径 ---
DINO_PATH = r"D:\backdoor2\picture\grounding-dino-base"
SAM_PATH = r"D:\backdoor2\picture\sam-vit-base"
VITMATTE_PATH = r"D:\backdoor2\picture\vitmatte-small-composition-1k"

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
MORPH_KERNEL_SIZE = 15

COLOR_PRIORS = ["red", "green", "blue", "yellow", "orange", "purple", "brown", "black", "white", "gray", "cyan"]
VLM_JSON_CACHE: Dict[str, str] = {}
dino_processor = dino_model = sam_processor = sam_model = vitmatte_processor = vitmatte_model = None


# ==========================================
# 2. 动态显存调度机制
# ==========================================
def load_ood_models():
    global dino_processor, dino_model, sam_processor, sam_model, vitmatte_processor, vitmatte_model
    if dino_model is not None: return
    print("\n正在加载 OOD 视觉大模型 (DINO + SAM + ViTMatte) 以进行精准定位与分割...")
    dino_processor = AutoProcessor.from_pretrained(DINO_PATH, local_files_only=True)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_PATH, local_files_only=True).to(DEVICE)
    sam_processor = SamProcessor.from_pretrained(SAM_PATH, local_files_only=True)
    sam_model = SamModel.from_pretrained(SAM_PATH, local_files_only=True).to(DEVICE)
    vitmatte_processor = VitMatteImageProcessor.from_pretrained(VITMATTE_PATH, local_files_only=True)
    vitmatte_model = VitMatteForImageMatting.from_pretrained(VITMATTE_PATH, local_files_only=True).to(DEVICE)
    print("大模型加载完毕！")


def unload_ood_models():
    global dino_processor, dino_model, sam_processor, sam_model, vitmatte_processor, vitmatte_model
    del dino_processor, dino_model, sam_processor, sam_model, vitmatte_processor, vitmatte_model
    dino_processor = dino_model = sam_processor = sam_model = vitmatte_processor = vitmatte_model = None
    torch.cuda.empty_cache()


# ==========================================
# 3. 数据集解析
# ==========================================
class LabelManager:
    def __init__(self):
        self.l2i, self.i2l = {}, {}

    def fit(self, labels):
        unique_labels = sorted(list(set([str(x) for x in labels])))
        self.l2i = {l: i for i, l in enumerate(unique_labels)}
        self.i2l = {i: l for l, i in self.l2i.items()}
        return len(unique_labels)

    def get_idx(self, label): return self.l2i[str(label)]

    def get_all_classes(self): return list(self.l2i.keys())


class DatasetParser:
    @staticmethod
    def parse(root: str, label_file: str) -> pd.DataFrame:
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"找不到标注文件: {label_file}")

        df = pd.read_csv(label_file)
        cols = df.columns.tolist()
        img_col = next((c for c in ["img_filename", "filename", "image", "path", "img_path"] if c in cols), cols[0])
        lbl_col = next((c for c in ["y", "label", "category", "class"] if c in cols), cols[1])

        records = []
        for _, row in df.iterrows():
            f_name = str(row[img_col]).strip()
            full_path = f_name if os.path.isabs(f_name) else os.path.join(root, f_name)

            if not os.path.exists(full_path):
                for sub in ["train", "val", "test", "images", ""]:
                    tmp = os.path.join(root, sub, f_name)
                    if os.path.exists(tmp):
                        full_path = tmp
                        break

            if os.path.exists(full_path):
                records.append({"img_path": os.path.abspath(full_path), "label": str(row[lbl_col]).strip()})

        parsed = pd.DataFrame(records)
        if parsed.empty:
            raise ValueError(f"解析失败！未找到图片文件！")
        return parsed.drop_duplicates(subset=["img_path"]).reset_index(drop=True)


# ==========================================
# 4. API 请求与基础工具
# ==========================================
def img_to_base64(img_np: np.ndarray, max_size: int = 512) -> str:
    img = Image.fromarray(img_np)
    img.thumbnail((max_size, max_size))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def request_qwen_vl(api_key: str, messages: List[dict], retries: int = 2) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": VLM_MODEL, "input": {"messages": [{"role": "user", "content": messages}]},
               "parameters": {"result_format": "message"}}
    for _ in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            if response.status_code != 200: continue
            content = response.json()["output"]["choices"][0]["message"]["content"]
            return content[0]["text"] if isinstance(content, list) else str(content)
        except Exception:
            continue
    return '{"color": "unknown"}'


def extract_json_block(text: str) -> dict:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match: return json.loads(match.group())
    except:
        pass
    return {"color": "unknown"}


# ==========================================
# 5. 统一图像分割引擎 (纯正 DINO + SAM)
# ==========================================
def extract_components(image_path: str, text_prompt: str) -> Optional[dict]:
    """完全统一物理分离流程，任何数据集均使用大模型进行定位和抠图"""
    try:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        orig_h, orig_w = image_np.shape[:2]

        inputs = dino_processor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = dino_model(**inputs)
        results = dino_processor.post_process_grounded_object_detection(
            outputs, input_ids=inputs.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]]
        )[0]
        if len(results["boxes"]) == 0: return None
        all_boxes = results["boxes"].cpu().numpy().tolist()

        inputs = sam_processor(image, input_boxes=[[all_boxes]], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = sam_model(**inputs)
        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )[0]
        combined_mask = np.zeros((orig_h, orig_w), dtype=bool)
        for i in range(len(all_boxes)):
            combined_mask = np.logical_or(combined_mask, masks[i][0].numpy())
        mask_uint8 = (combined_mask * 255).astype(np.uint8)

        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        trimap = np.zeros_like(mask_uint8)
        trimap[cv2.dilate(mask_uint8, kernel, iterations=2) == 255] = 128
        trimap[cv2.erode(mask_uint8, kernel, iterations=2) == 255] = 255

        inputs = vitmatte_processor(images=image, trimaps=Image.fromarray(trimap).convert("L"), return_tensors="pt").to(
            DEVICE)
        with torch.no_grad():
            alphas = vitmatte_model(**inputs).alphas
        alpha_matte = (alphas.squeeze().cpu().numpy() * 255).astype(np.uint8)[:orig_h, :orig_w]

        alpha_3d = np.stack([alpha_matte.astype(float) / 255.0] * 3, axis=2)
        fg_isolated = (image_np * alpha_3d + np.ones_like(image_np) * 255 * (1 - alpha_3d)).astype(np.uint8)
        bg_isolated = (image_np * (1 - alpha_3d)).astype(np.uint8)

        return {"orig_np": image_np, "fg_img": fg_isolated, "bg_img": bg_isolated, "mask": alpha_matte}

    except Exception as e:
        return None


# ==========================================
# 6. 纯净颜色挖掘 (VLM 盲测提取) 与因果差值 PBS
# ==========================================
def mine_color_vlm(api_key: str, img_np: np.ndarray, region_type: str) -> str:
    b64 = img_to_base64(img_np)
    cache_key = hashlib.md5((b64 + region_type).encode("utf-8")).hexdigest()
    if cache_key in VLM_JSON_CACHE: return VLM_JSON_CACHE[cache_key]

    priors_str = ", ".join(COLOR_PRIORS)
    if region_type == "fg":
        prompt = (f"这是一张背景处理为纯白色的图像。仅分析【非白色的中心主导区域】的主导颜色。"
                  f"严格选一个：[{priors_str}]。只返回 JSON：{{\"color\":\"候选颜色\"}}")
    else:
        prompt = (f"这是一张中心处理为黑色的图像。仅分析【外围背景区域】的主导颜色。"
                  f"严格选一个：[{priors_str}]。只返回 JSON：{{\"color\":\"候选颜色\"}}")

    text = request_qwen_vl(api_key, [{"image": f"data:image/jpeg;base64,{b64}"}, {"text": prompt}])
    obj = extract_json_block(text)
    color = str(obj.get("color", "unknown")).strip().lower()
    VLM_JSON_CACHE[cache_key] = color
    return color


def compute_spurious_correlation(df: pd.DataFrame, feature_col: str) -> float:
    pbs_list = []
    unique_labels = df["label"].unique()
    for lbl in unique_labels:
        in_cls = df[df["label"] == lbl]
        out_cls = df[df["label"] != lbl]
        if in_cls.empty: continue
        counts = in_cls[feature_col].value_counts(normalize=True)
        if counts.empty: continue
        dom_color = counts.index[0]
        p_in = counts.iloc[0]
        p_out = (out_cls[feature_col] == dom_color).mean() if not out_cls.empty else 0.0
        pbs_list.append(max(0.0, float(p_in - p_out)))
    return float(np.mean(pbs_list))


def compute_color_pbs(api_key: str, train_df: pd.DataFrame, dino_prompt: str) -> dict:
    sampled_df = train_df.groupby("label").sample(
        n=min(int(AUDIT_SAMPLE_COUNT_TRAIN / train_df["label"].nunique()), len(train_df)), random_state=SEED)
    rows = []
    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="执行物理分离与纯净颜色 VLM 挖掘"):
        comp = extract_components(row["img_path"], dino_prompt)
        if not comp: continue
        try:
            fg_color = mine_color_vlm(api_key, comp["fg_img"], "fg")
            bg_color = mine_color_vlm(api_key, comp["bg_img"], "bg")
            rows.append(
                {"img_path": row["img_path"], "label": row["label"], "fg_color": fg_color, "bg_color": bg_color})
        except Exception as e:
            continue
    if not rows: raise RuntimeError("颜色特征挖掘失败。")
    detail_df = pd.DataFrame(rows)
    return {
        "pbs_fg": compute_spurious_correlation(detail_df, "fg_color"),
        "pbs_bg": compute_spurious_correlation(detail_df, "bg_color")
    }


# ==========================================
# 7. OOD 绝杀：类别色彩对立移植与绝对亮度拉平
# ==========================================
def equalize_luminance_and_grayscale(img_np: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    mean, std = cv2.meanStdDev(L)
    std_val = std[0][0] if std[0][0] > 0 else 1.0
    L_new = ((L - mean[0][0]) * (64.0 / std_val)) + 128.0
    L_new = np.clip(L_new, 0, 255)

    lab[:, :, 0] = L_new
    lab[:, :, 1] = 128.0
    lab[:, :, 2] = 128.0
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def reinhard_color_transfer(src_rgb: np.ndarray, dst_rgb: np.ndarray, src_mask: np.ndarray,
                            dst_mask: np.ndarray) -> np.ndarray:
    src_lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    if cv2.countNonZero(src_mask) == 0 or cv2.countNonZero(dst_mask) == 0:
        return src_rgb.copy()

    s_mean, s_std = cv2.meanStdDev(src_lab, mask=src_mask)
    d_mean, d_std = cv2.meanStdDev(dst_lab, mask=dst_mask)
    s_std = np.where(s_std == 0, 1.0, s_std)

    result_lab = src_lab.copy()
    for i in range(3):
        result_lab[:, :, i] = ((result_lab[:, :, i] - s_mean[i][0]) * (d_std[i][0] / s_std[i][0])) + d_mean[i][0]

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)

    final = src_rgb.copy()
    mask_bool = src_mask > 128
    final[mask_bool] = result_rgb[mask_bool]
    return final


def build_donor_bank(train_df: pd.DataFrame, dino_prompt: str, samples_per_class=5) -> dict:
    donor_bank = {}
    print("\n[准备] 正在从训练集中提取各类别精准颜色特征库...")
    for lbl, group in train_df.groupby("label"):
        bank = []
        sampled = group.sample(n=min(samples_per_class, len(group)), random_state=SEED)
        for _, row in sampled.iterrows():
            comp = extract_components(row["img_path"], dino_prompt)
            if comp: bank.append(comp)
        donor_bank[lbl] = bank
    return donor_bank


def build_specific_color_ood(test_df: pd.DataFrame, train_df: pd.DataFrame, dino_prompt: str, target_n: int, mode: str,
                             save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    sampled = test_df.sample(n=min(target_n, len(test_df)), random_state=SEED)
    ood_records, fail_count = [], 0

    donor_bank = {}
    unique_labels = []
    if "color_swap" in mode:
        donor_bank = build_donor_bank(train_df, dino_prompt)
        unique_labels = list(donor_bank.keys())

    for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc=f"构造 OOD 集合: {mode}"):
        comp = extract_components(row["img_path"], dino_prompt)
        if not comp:
            fail_count += 1;
            continue

        img_np, mask = comp["orig_np"], comp["mask"]
        lbl = row["label"]

        if mode == "grayscale_norm":
            ood_img = equalize_luminance_and_grayscale(img_np)

        elif "color_swap" in mode:
            target_lbls = [l for l in unique_labels if l != lbl]
            target_lbl = random.choice(target_lbls) if target_lbls else lbl
            donors = donor_bank.get(target_lbl, [])
            if not donors:
                fail_count += 1;
                continue
            donor = random.choice(donors)

            if mode == "fg_color_swap":
                ood_img = reinhard_color_transfer(img_np, donor["orig_np"], mask, donor["mask"])
            elif mode == "bg_color_swap":
                inv_mask = 255 - mask
                inv_donor_mask = 255 - donor["mask"]
                ood_img = reinhard_color_transfer(img_np, donor["orig_np"], inv_mask, inv_donor_mask)
            elif mode == "full_color_swap":
                inv_mask = 255 - mask
                inv_donor_mask = 255 - donor["mask"]
                tmp = reinhard_color_transfer(img_np, donor["orig_np"], mask, donor["mask"])
                ood_img = reinhard_color_transfer(tmp, donor["orig_np"], inv_mask, inv_donor_mask)
        else:
            ood_img = img_np

        save_name = f"ood_{mode}_{lbl}_{len(ood_records):04d}.jpg"
        final_path = os.path.join(save_dir, save_name)
        Image.fromarray(ood_img).save(final_path, quality=95)
        ood_records.append({"img_path": final_path, "label": lbl})

    return pd.DataFrame(ood_records), fail_count


# ==========================================
# 8. 模型训练与评估
# ==========================================
class AuditDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, lm: LabelManager):
        self.df, self.transform, self.lm = df.reset_index(drop=True), transform, lm

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return self.transform(Image.open(row["img_path"]).convert("RGB")), self.lm.get_idx(row["label"])


# def build_resnet_model(num_classes: int):
#     model = models.resnet18(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model.to(DEVICE)
#
#
# def train_model(model, train_df: pd.DataFrame, lm: LabelManager):
#     train_tf = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ])
#     loader = DataLoader(AuditDataset(train_df, train_tf, lm), batch_size=BATCH_SIZE, shuffle=True)
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     criterion = nn.CrossEntropyLoss()

def build_resnet_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters(): param.requires_grad = False
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes))
    return model.to(DEVICE)

def train_model(model, train_df: pd.DataFrame, lm: LabelManager):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(AuditDataset(train_df, train_tf, lm), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += labels.size(0)
        print(f"  Epoch {epoch + 1:02d} | Loss: {epoch_loss / len(loader):.4f} | Acc: {correct / total * 100:.2f}%")


def evaluate_model(model, df: pd.DataFrame, lm: LabelManager) -> dict:
    model.eval()
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if df.empty: return {"pretty": "0.00%", "acc": 0.0}
    loader = DataLoader(AuditDataset(df, eval_tf, lm), batch_size=32, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (torch.argmax(model(x), dim=1) == y).sum().item()
            total += y.size(0)
    acc = correct / max(1, total)
    return {"pretty": f"{acc * 100:.2f}%", "acc": acc}


# ==========================================
# 9. 主函数
# ==========================================
def main():
    print("\n" + "严谨因果版：统一物理分离 + VLM 色彩挖掘框架".center(60, "="))
    api_key = input("0. DashScope API Key: ").strip()
    root = input("1. 数据集根目录: ").strip()
    csv_p = input("2. 标注CSV路径: ").strip()
    ds_name = input("3. 数据集简称: ").strip()
    expert = input("4. 先验知识 (分类主体, 用于DINO定位): ").strip()

    if os.path.exists(RESULTS_DIR): shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    full_df = DatasetParser.parse(root, csv_p)
    train_df, test_df = train_test_split(full_df, test_size=0.2, stratify=full_df["label"], random_state=SEED)
    lm = LabelManager()
    num_classes = lm.fit(full_df["label"])

    dino_prompt = f"a {expert}."

    # 彻底弃用任何所谓“针对数字的优化分流”，无条件加载大模型引擎！
    load_ood_models()

    print("\n[Step 1] 执行统一物理分离 (定位) 与 VLM 纯净颜色挖掘...")
    pbs_info = compute_color_pbs(api_key, train_df, dino_prompt)
    print(f"主体颜色偏见 (FG_PBS): {pbs_info['pbs_fg']:.4f}")
    print(f"背景颜色偏见 (BG_PBS): {pbs_info['pbs_bg']:.4f}")

    ood_modes = {}
    print("\n[判定] 正在基于阈值 (0.5) 决定真正的对立 OOD 构造策略...")

    if pbs_info['pbs_fg'] > SHORTCUT_THRESHOLD or pbs_info['pbs_bg'] > SHORTCUT_THRESHOLD:

        ood_modes["grayscale_norm"] = "OOD_Grayscale_Norm"
        print(" -> 检测到捷径，启用【绝对亮度拉平+全图灰度化】严格测试。")

        if pbs_info['pbs_fg'] > SHORTCUT_THRESHOLD:
            ood_modes["fg_color_swap"] = "OOD_FG_Color_Swap"
            print(" -> [致命干预] 主体存在捷径，启用【对立类别特征移植(主体)】！")

        if pbs_info['pbs_bg'] > SHORTCUT_THRESHOLD:
            ood_modes["bg_color_swap"] = "OOD_BG_Color_Swap"
            print(" -> [致命干预] 背景存在捷径，启用【对立类别特征移植(背景)】！")

        if pbs_info['pbs_fg'] > SHORTCUT_THRESHOLD and pbs_info['pbs_bg'] > SHORTCUT_THRESHOLD:
            ood_modes["full_color_swap"] = "OOD_Full_Color_Swap"
            print(" -> [终极干预] 全局捷径，启用【主体+背景双重对立色彩移植】！")
    else:
        print(" -> 未检测到光学偏见。")

    ood_datasets = {}
    if ood_modes:
        print("\n[Step 2] 正在构建靶向 OOD 测试集...")
        for mode, dir_name in ood_modes.items():
            save_dir = os.path.join(RESULTS_DIR, dir_name)
            ood_df, fail_count = build_specific_color_ood(test_df, train_df, dino_prompt, OOD_TARGET_COUNT, mode,
                                                          save_dir)
            ood_datasets[mode] = {"df": ood_df, "fail_count": fail_count}
            print(f"  >> [{mode}] 构造完成: {len(ood_df)} 张 | 失败: {fail_count}")

    unload_ood_models()

    print(f"\n[Step 3] 训练强特征提取基线模型 (ResNet-18)...")
    model = build_resnet_model(num_classes)
    train_model(model, train_df, lm)

    print("\n[Step 4] 评估因果断层 (绝对色彩干预 vs 绝对亮度拉平)...")
    full_eval = evaluate_model(model, test_df, lm)

    report_lines = [
        "=" * 60,
        "对立靶向光学捷径因果审计报告".center(52),
        "=" * 60,
        f"数据集: {ds_name}",
        f"1. 主体色彩关联强度 (FG_PBS): {pbs_info['pbs_fg']:.4f}",
        f"2. 背景色彩关联强度 (BG_PBS): {pbs_info['pbs_bg']:.4f}",
        "-" * 60,
        f"[基准] 原生测试准确率 (ID):     {full_eval['pretty']}"
    ]

    if ood_modes:
        for mode in ood_modes.keys():
            ood_eval = evaluate_model(model, ood_datasets[mode]["df"], lm)
            drop_rate = (full_eval['acc'] - ood_eval['acc']) * 100
            report_lines.append(f"[{mode}] 干预后准确率:    {ood_eval['pretty']} (因果掉点: {drop_rate:.2f}%)")

    report_lines.append("=" * 60)
    report = "\n".join(report_lines)

    print("\n" + report)
    with open(os.path.join(RESULTS_DIR, f"strict_color_report_{ds_name}.txt"), "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()