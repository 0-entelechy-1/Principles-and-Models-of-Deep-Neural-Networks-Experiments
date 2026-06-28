import os, sys, time, json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
from PIL import Image
import urllib.request

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

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
    'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ============================================================
# 1. 下载真实测试图像 (COCO val2017)
# ============================================================
log("=" * 60)
log("1. 下载真实测试图像 (COCO val2017)")
log("=" * 60)

SAMPLE_DIR = os.path.join(OUTPUT_DIR, 'real_images')
os.makedirs(SAMPLE_DIR, exist_ok=True)

# 固定的 COCO val2017 图像 ID，包含常见目标（人、车辆、动物等）
COCO_IMAGE_IDS = [
    '000000000139', '000000000285', '000000000632',
    '000000000724', '000000000776', '000000000785'
]

def download_coco_image(img_id):
    url = f'http://images.cocodataset.org/val2017/{img_id}.jpg'
    path = os.path.join(SAMPLE_DIR, f'{img_id}.jpg')
    if not os.path.exists(path):
        log(f"  下载: {img_id}.jpg ...")
        urllib.request.urlretrieve(url, path)
    else:
        log(f"  已存在: {img_id}.jpg")
    return path

sample_images = []
for img_id in COCO_IMAGE_IDS:
    path = download_coco_image(img_id)
    sample_images.append(path)
    log(f"  准备完成: {os.path.basename(path)} (大小 {os.path.getsize(path)} 字节)")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for idx, path in enumerate(sample_images):
    r, c = idx // 3, idx % 3
    img = Image.open(path)
    axes[r, c].imshow(img)
    axes[r, c].set_title(f'COCO测试图 {idx+1}', fontsize=10)
    axes[r, c].axis('off')
fig.suptitle('目标检测真实测试图像 (COCO val2017)', fontsize=14)
save_fig(fig, 'test_images.png')

# ============================================================
# 2. Faster R-CNN (二阶段)
# ============================================================
log("\n" + "=" * 60)
log("2. Faster R-CNN (二阶段目标检测)")
log("=" * 60)

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

log("加载 Faster R-CNN (ResNet50-FPN)...")
frcnn = fasterrcnn_resnet50_fpn(weights='DEFAULT')
frcnn.eval()

n_frcnn = sum(p.numel() for p in frcnn.parameters())
log(f"  参数量: {n_frcnn:,}")

frcnn_results = []
for path in sample_images:
    img = Image.open(path).convert('RGB')
    img_t = torchvision.transforms.functional.to_tensor(img).unsqueeze(0)
    t0 = time.time()
    with torch.no_grad():
        preds = frcnn(img_t)
    elapsed = time.time() - t0
    pred = preds[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    frcnn_results.append({
        'img': img, 'boxes': boxes, 'scores': scores, 'labels': labels,
        'time': elapsed, 'path': path})
    n_det = (scores > 0.5).sum()
    log(f"  {os.path.basename(path)}: {n_det} 检测 (阈值>0.5), 耗时 {elapsed:.2f}s")

# ============================================================
# 3. YOLO (一阶段)
# ============================================================
log("\n" + "=" * 60)
log("3. YOLO (一阶段目标检测)")
log("=" * 60)

from ultralytics import YOLO

yolo_path = r'C:\Users\29146\.cache\YOLO\yolo26s.pt'
log(f"加载 YOLO: {yolo_path}")
yolo = YOLO(yolo_path)

n_yolo = sum(p.numel() for p in yolo.model.parameters())
log(f"  参数量: {n_yolo:,}")

yolo_results = []
for path in sample_images:
    t0 = time.time()
    results = yolo.predict(path, verbose=False, conf=0.25)
    elapsed = time.time() - t0
    r = results[0]
    yolo_results.append({
        'img': Image.open(path).convert('RGB'),
        'boxes': r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.array([]),
        'scores': r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else np.array([]),
        'labels': r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) > 0 else np.array([]),
        'time': elapsed, 'path': path,
        'result_obj': r})
    n_det = len(r.boxes)
    log(f"  {os.path.basename(path)}: {n_det} 检测, 耗时 {elapsed:.2f}s")

# ============================================================
# 4. 可视化对比
# ============================================================
log("\n" + "=" * 60)
log("4. 检测结果可视化对比")
log("=" * 60)

COLORS = plt.cm.tab20(np.linspace(0, 1, 20))

def draw_detections(ax, img, boxes, scores, labels, class_names, threshold=0.3, title=''):
    ax.imshow(img)
    for i in range(len(boxes)):
        if scores[i] < threshold:
            continue
        box = boxes[i]
        color = COLORS[int(labels[i]) % 20]
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                  linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        label_text = class_names[labels[i]] if labels[i] < len(class_names) else f'cls{labels[i]}'
        ax.text(box[0], box[1]-5, f'{label_text} {scores[i]:.2f}',
                fontsize=7, color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color[:3], alpha=0.7))
    ax.set_title(title, fontsize=11)
    ax.axis('off')

n_samples = len(sample_images)
fig, axes = plt.subplots(n_samples, 2, figsize=(14, 5*n_samples))
if n_samples == 1:
    axes = axes.reshape(1, -1)

for i in range(n_samples):
    fr = frcnn_results[i]
    draw_detections(axes[i, 0], fr['img'], fr['boxes'], fr['scores'],
                    fr['labels'], COCO_CLASSES, threshold=0.3,
                    title=f'Faster R-CNN ({fr["time"]:.2f}s)')
    yr = yolo_results[i]
    draw_detections(axes[i, 1], yr['img'], yr['boxes'], yr['scores'],
                    yr['labels'], yolo.names if hasattr(yolo, 'names') and isinstance(yolo.names, list) else COCO_CLASSES,
                    threshold=0.25,
                    title=f'YOLO ({yr["time"]:.2f}s)')

fig.suptitle('Faster R-CNN vs YOLO 检测结果对比', fontsize=14, y=1.01)
save_fig(fig, 'detection_comparison.png')

# 单独每张模型的详细结果
for model_name, results_list in [('frcnn', frcnn_results), ('yolo', yolo_results)]:
    for i, r in enumerate(results_list):
        fig, ax = plt.subplots(figsize=(10, 7))
        cn = COCO_CLASSES if model_name == 'frcnn' else (yolo.names if hasattr(yolo, 'names') and isinstance(yolo.names, list) else COCO_CLASSES)
        draw_detections(ax, r['img'], r['boxes'], r['scores'],
                        r['labels'], cn, threshold=0.25,
                        title=f'{model_name.upper()} — {os.path.basename(r["path"])}')
        save_fig(fig, f'detection_{model_name}_{i+1}.png')

# ============================================================
# 5. 结构对比分析
# ============================================================
log("\n" + "=" * 60)
log("5. 模型结构对比分析")
log("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 参数量对比
names = ['Faster R-CNN', 'YOLO']
params_list = [n_frcnn, n_yolo]
axes[0].bar(names, params_list, color=['#3498db', '#e74c3c'])
axes[0].set_ylabel('参数量')
axes[0].set_title('模型参数量对比')
for i, v in enumerate(params_list):
    axes[0].text(i, v + max(params_list)*0.02, f'{v:,}', ha='center', fontsize=10)

# 推理速度对比
avg_frcnn = np.mean([r['time'] for r in frcnn_results])
avg_yolo = np.mean([r['time'] for r in yolo_results])
axes[1].bar(names, [avg_frcnn, avg_yolo], color=['#3498db', '#e74c3c'])
axes[1].set_ylabel('平均推理时间 (秒)')
axes[1].set_title('CPU推理速度对比')
for i, v in enumerate([avg_frcnn, avg_yolo]):
    axes[1].text(i, v + 0.05, f'{v:.2f}s', ha='center', fontsize=10)

save_fig(fig, 'model_comparison.png')

# 检测数量统计
fig, ax = plt.subplots(figsize=(10, 5))
frcnn_dets = [(r['scores'] > 0.5).sum() for r in frcnn_results]
yolo_dets = [len(r['boxes']) for r in yolo_results]
x = np.arange(n_samples)
w = 0.35
ax.bar(x - w/2, frcnn_dets, w, label='Faster R-CNN (conf>0.5)', color='#3498db')
ax.bar(x + w/2, yolo_dets, w, label='YOLO (conf>0.25)', color='#e74c3c')
ax.set_xlabel('测试图像')
ax.set_ylabel('检测数量')
ax.set_title('各图像检测数量对比')
ax.set_xticks(x)
ax.set_xticklabels([f'图{i+1}' for i in range(n_samples)])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
save_fig(fig, 'detection_count_comparison.png')

# ============================================================
# 6. 汇总
# ============================================================
log("\n" + "=" * 60)
log("实验汇总")
log("=" * 60)
log(f"\n--- 模型对比 ---")
log(f"  Faster R-CNN: 参数量={n_frcnn:,}, 平均推理时间={avg_frcnn:.2f}s")
log(f"  YOLO:         参数量={n_yolo:,}, 平均推理时间={avg_yolo:.2f}s")
log(f"\n--- 检测统计 ---")
for i in range(n_samples):
    log(f"  图{i+1}: Faster R-CNN {frcnn_dets[i]}个, YOLO {yolo_dets[i]}个")

log("\n实验完成!")
log_fh.close()
