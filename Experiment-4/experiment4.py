import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
from PIL import Image, ImageDraw
import cv2

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, 'experiment_log.txt')
log_fh = open(LOG_FILE, 'w', encoding='utf-8')

def log(msg):
    print(msg); log_fh.write(str(msg) + '\n'); log_fh.flush()

def save_fig(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name), bbox_inches='tight')
    plt.close(fig)
    log(f"  [saved] {name}")

device = torch.device('cpu')
log(f"设备: {device}")

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tvmonitor'
]

PALETTE = []
for i in range(21):
    r, g, b = 0, 0, 0
    c = i
    for j in range(8):
        r |= ((c >> 0) & 1) << (7 - j)
        g |= ((c >> 1) & 1) << (7 - j)
        b |= ((c >> 2) & 1) << (7 - j)
        c >>= 3
    PALETTE.append([r, g, b])
PALETTE = np.array(PALETTE)
# 背景类不再使用纯黑，避免可视化时与“无输出”混淆
PALETTE[0] = [64, 64, 64]

# 预训练模型使用的 ImageNet 归一化
from torchvision import transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# ============================================================
# 1. 生成/加载测试图像
# ============================================================
log("=" * 60)
log("1. 加载测试图像")
log("=" * 60)

SAMPLE_DIR = os.path.join(OUTPUT_DIR, 'samples')
os.makedirs(SAMPLE_DIR, exist_ok=True)
REAL_IMG_DIR = os.path.join(OUTPUT_DIR, 'real_images')

def create_segmentation_scene(filename, w=512, h=384):
    img = Image.new('RGB', (w, h), (135, 206, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, int(h*0.6), w, h], fill=(34, 139, 34))
    draw.rectangle([int(w*0.1), int(h*0.3), int(w*0.3), int(h*0.6)], fill=(139, 69, 19))
    draw.rectangle([int(w*0.5), int(h*0.2), int(w*0.8), int(h*0.55)], fill=(128, 128, 128))
    draw.ellipse([int(w*0.35), int(h*0.1), int(w*0.5), int(h*0.25)], fill=(255, 255, 0))
    draw.rectangle([int(w*0.6), int(h*0.55), int(w*0.75), int(h*0.8)], fill=(255, 0, 0))
    path = os.path.join(SAMPLE_DIR, filename)
    img.save(path)
    return img, path

sample_images = []

# 优先使用真实场景图像
if os.path.isdir(REAL_IMG_DIR):
    real_paths = sorted([
        os.path.join(REAL_IMG_DIR, f)
        for f in os.listdir(REAL_IMG_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])
    sample_images.extend(real_paths[:6])
    log(f"  加载真实图像: {len(real_paths[:6])} 张")

# 若真实图像不足，用 1 张合成图补充
if len(sample_images) < 2:
    _, path = create_segmentation_scene('scene_fallback.jpg')
    sample_images.append(path)
    log(f"  创建合成补充图像: scene_fallback.jpg")

for p in sample_images:
    log(f"  使用: {os.path.basename(p)}")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for idx, path in enumerate(sample_images):
    r, c = idx // 3, idx % 3
    img = Image.open(path)
    axes[r, c].imshow(img)
    axes[r, c].set_title(f'测试图 {idx+1}', fontsize=10)
    axes[r, c].axis('off')
# 隐藏多余子图
for idx in range(len(sample_images), 6):
    r, c = idx // 3, idx % 3
    axes[r, c].axis('off')
fig.suptitle('图像分割测试图像', fontsize=14)
save_fig(fig, 'test_images.png')

# ============================================================
# 2. FCN-ResNet50
# ============================================================
log("\n" + "=" * 60)
log("2. FCN-ResNet50 (全卷积网络)")
log("=" * 60)

import torchvision
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50

log("加载 FCN-ResNet50 (预训练)...")
fcn = fcn_resnet50(weights='DEFAULT')
fcn.eval()
n_fcn = sum(p.numel() for p in fcn.parameters())
log(f"  参数量: {n_fcn:,}")

fcn_results = []
for path in sample_images:
    img = Image.open(path).convert('RGB')
    img_t = normalize(torchvision.transforms.functional.to_tensor(img)).unsqueeze(0)
    t0 = time.time()
    with torch.no_grad():
        out = fcn(img_t)['out']
    elapsed = time.time() - t0
    seg = out.argmax(1).squeeze(0).cpu().numpy()
    fcn_results.append({'img': img, 'seg': seg, 'time': elapsed, 'path': path,
                        'n_classes': len(np.unique(seg))})
    log(f"  {os.path.basename(path)}: {len(np.unique(seg))} 类, 耗时 {elapsed:.2f}s")

# ============================================================
# 3. DeepLabV3-ResNet50
# ============================================================
log("\n" + "=" * 60)
log("3. DeepLabV3-ResNet50")
log("=" * 60)

log("加载 DeepLabV3-ResNet50 (预训练)...")
deeplab = deeplabv3_resnet50(weights='DEFAULT')
deeplab.eval()
n_dl = sum(p.numel() for p in deeplab.parameters())
log(f"  参数量: {n_dl:,}")

dl_results = []
for path in sample_images:
    img = Image.open(path).convert('RGB')
    img_t = normalize(torchvision.transforms.functional.to_tensor(img)).unsqueeze(0)
    t0 = time.time()
    with torch.no_grad():
        out = deeplab(img_t)['out']
    elapsed = time.time() - t0
    seg = out.argmax(1).squeeze(0).cpu().numpy()
    dl_results.append({'img': img, 'seg': seg, 'time': elapsed, 'path': path,
                       'n_classes': len(np.unique(seg))})
    log(f"  {os.path.basename(path)}: {len(np.unique(seg))} 类, 耗时 {elapsed:.2f}s")

# ============================================================
# 4. 可视化
# ============================================================
log("\n" + "=" * 60)
log("4. 分割结果可视化")
log("=" * 60)

def colorize_seg(seg, palette):
    h, w = seg.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(palette)):
        mask = seg == c
        color_img[mask] = palette[c]
    return color_img

def overlay_seg(img, seg, palette, alpha=0.5):
    img_np = np.array(img)
    color_seg = colorize_seg(seg, palette)
    h, w = min(img_np.shape[0], color_seg.shape[0]), min(img_np.shape[1], color_seg.shape[1])
    blended = img_np[:h, :w].copy()
    mask = color_seg[:h, :w].sum(axis=2) > 0
    blended[mask] = (alpha * color_seg[:h, :w][mask] + (1-alpha) * img_np[:h, :w][mask]).astype(np.uint8)
    return blended

def add_class_legend(ax, seg, palette, classes, loc='lower right', fontsize=7):
    """在 ax 右下角添加实际出现类别的图例。"""
    unique_cls = sorted(np.unique(seg).tolist())
    handles = []
    for c in unique_cls:
        if 0 <= c < len(classes):
            color = palette[c] / 255.0
            handles.append(Patch(facecolor=color, edgecolor='k', label=classes[c]))
    if handles:
        ax.legend(handles=handles, loc=loc, fontsize=fontsize,
                  frameon=True, fancybox=True, shadow=False)

n_samples = len(sample_images)
fig, axes = plt.subplots(n_samples, 3, figsize=(16, 5*n_samples))
if n_samples == 1:
    axes = axes.reshape(1, -1)

for i in range(n_samples):
    fr = fcn_results[i]
    dr = dl_results[i]
    axes[i, 0].imshow(fr['img'])
    axes[i, 0].set_title(f'原图: {os.path.basename(fr["path"])}', fontsize=10)
    axes[i, 0].axis('off')

    fcn_overlay = overlay_seg(fr['img'], fr['seg'], PALETTE)
    axes[i, 1].imshow(fcn_overlay)
    axes[i, 1].set_title(f'FCN ({fr["time"]:.2f}s, {fr["n_classes"]}类)', fontsize=10)
    axes[i, 1].axis('off')
    add_class_legend(axes[i, 1], fr['seg'], PALETTE, VOC_CLASSES)

    dl_overlay = overlay_seg(dr['img'], dr['seg'], PALETTE)
    axes[i, 2].imshow(dl_overlay)
    axes[i, 2].set_title(f'DeepLabV3 ({dr["time"]:.2f}s, {dr["n_classes"]}类)', fontsize=10)
    axes[i, 2].axis('off')
    add_class_legend(axes[i, 2], dr['seg'], PALETTE, VOC_CLASSES)

fig.suptitle('FCN vs DeepLabV3 分割结果对比', fontsize=14, y=1.01)
save_fig(fig, 'segmentation_comparison.png')

# 单独每张详细结果
for i in range(n_samples):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fr, dr = fcn_results[i], dl_results[i]

    axes[0, 0].imshow(fr['img'])
    axes[0, 0].set_title('原图')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(colorize_seg(fr['seg'], PALETTE))
    axes[0, 1].set_title(f'FCN 分割图 ({fr["n_classes"]}类)')
    axes[0, 1].axis('off')
    add_class_legend(axes[0, 1], fr['seg'], PALETTE, VOC_CLASSES)

    axes[0, 2].imshow(overlay_seg(fr['img'], fr['seg'], PALETTE))
    axes[0, 2].set_title('FCN 叠加结果')
    axes[0, 2].axis('off')
    add_class_legend(axes[0, 2], fr['seg'], PALETTE, VOC_CLASSES)

    axes[1, 0].imshow(dr['img'])
    axes[1, 0].set_title('原图')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(colorize_seg(dr['seg'], PALETTE))
    axes[1, 1].set_title(f'DeepLabV3 分割图 ({dr["n_classes"]}类)')
    axes[1, 1].axis('off')
    add_class_legend(axes[1, 1], dr['seg'], PALETTE, VOC_CLASSES)

    axes[1, 2].imshow(overlay_seg(dr['img'], dr['seg'], PALETTE))
    axes[1, 2].set_title('DeepLabV3 叠加结果')
    axes[1, 2].axis('off')
    add_class_legend(axes[1, 2], dr['seg'], PALETTE, VOC_CLASSES)

    fig.suptitle(f'{os.path.basename(sample_images[i])} — 分割详细结果', fontsize=13)
    save_fig(fig, f'segmentation_detail_{i+1}.png')

# ============================================================
# 5. 模型对比分析
# ============================================================
log("\n" + "=" * 60)
log("5. 模型对比分析")
log("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 参数量
names = ['FCN-ResNet50', 'DeepLabV3-ResNet50']
params_list = [n_fcn, n_dl]
axes[0].bar(names, params_list, color=['#3498db', '#e74c3c'])
axes[0].set_ylabel('参数量')
axes[0].set_title('模型参数量对比')
for i, v in enumerate(params_list):
    axes[0].text(i, v + max(params_list)*0.02, f'{v:,}', ha='center', fontsize=10)

# 推理速度
avg_fcn = np.mean([r['time'] for r in fcn_results])
avg_dl = np.mean([r['time'] for r in dl_results])
axes[1].bar(names, [avg_fcn, avg_dl], color=['#3498db', '#e74c3c'])
axes[1].set_ylabel('平均推理时间 (秒)')
axes[1].set_title('CPU推理速度对比')
for i, v in enumerate([avg_fcn, avg_dl]):
    axes[1].text(i, v + 0.05, f'{v:.2f}s', ha='center', fontsize=10)

# 检测到的类别数
fcn_avg_cls = np.mean([r['n_classes'] for r in fcn_results])
dl_avg_cls = np.mean([r['n_classes'] for r in dl_results])
axes[2].bar(names, [fcn_avg_cls, dl_avg_cls], color=['#3498db', '#e74c3c'])
axes[2].set_ylabel('平均检测类别数')
axes[2].set_title('分割类别数对比')
for i, v in enumerate([fcn_avg_cls, dl_avg_cls]):
    axes[2].text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=10)

save_fig(fig, 'model_comparison.png')

# IoU 相似度对比 (两个模型分割结果的一致性)
fig, ax = plt.subplots(figsize=(10, 5))
ious = []
for i in range(n_samples):
    fcn_seg = fcn_results[i]['seg']
    dl_seg = dl_results[i]['seg']
    h, w = min(fcn_seg.shape[0], dl_seg.shape[0]), min(fcn_seg.shape[1], dl_seg.shape[1])
    f, d = fcn_seg[:h, :w], dl_seg[:h, :w]
    intersection = (f == d).sum()
    total = h * w
    iou = intersection / total
    ious.append(iou)

ax.bar([f'图{i+1}' for i in range(n_samples)], ious, color='#2ecc71')
ax.set_ylabel('像素一致性')
ax.set_title('FCN vs DeepLabV3 分割结果像素一致性')
ax.set_ylim(0, 1)
for i, v in enumerate(ious):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
save_fig(fig, 'pixel_agreement.png')

# ============================================================
# 6. 汇总
# ============================================================
log("\n" + "=" * 60)
log("实验汇总")
log("=" * 60)
log(f"\n--- 模型对比 ---")
log(f"  FCN-ResNet50:     参数量={n_fcn:,}, 平均推理时间={avg_fcn:.2f}s, 平均类别数={fcn_avg_cls:.1f}")
log(f"  DeepLabV3-ResNet50: 参数量={n_dl:,}, 平均推理时间={avg_dl:.2f}s, 平均类别数={dl_avg_cls:.1f}")
log(f"\n--- 像素一致性 ---")
for i, iou in enumerate(ious):
    log(f"  图{i+1}: {iou:.3f}")

log("\n实验完成!")
log_fh.close()
