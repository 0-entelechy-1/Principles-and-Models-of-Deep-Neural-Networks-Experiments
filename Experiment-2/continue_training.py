"""
续训脚本：跳过已完成的MLP和VGG训练，只训练ResNet-18和子实验
然后生成所有可视化图片和报告
"""
import os, sys, time, pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_DIR = os.path.join(BASE_DIR, 'data')

LOG_FILE = os.path.join(OUTPUT_DIR, 'experiment_log.txt')

# 读取已有的log内容
existing_log = ''
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        existing_log = f.read()

log_fh = open(LOG_FILE, 'w', encoding='utf-8')
# 写入已有内容
log_fh.write(existing_log)
log_fh.flush()

def log(msg):
    print(msg); log_fh.write(str(msg) + '\n'); log_fh.flush()

def save_fig(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name), bbox_inches='tight')
    plt.close(fig)
    log(f"  [saved] {name}")

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cpu')

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 数据集
log("\n[续训] 加载CIFAR-10...")
transform_train = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=False, transform=transform_test)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

# ============================================================
# 模型定义 (与experiment2.py相同)
# ============================================================
class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3*32*32, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 10))
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

class VGGSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 10))
    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make(64, 64, 2, 1)
        self.layer2 = self._make(64, 128, 2, 2)
        self.layer3 = self._make(128, 256, 2, 2)
        self.layer4 = self._make(256, 512, 2, 2)
        self.fc = nn.Linear(512, 10)
    def _make(self, in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
        return self.fc(out)

class SEBlock(nn.Module):
    def __init__(self, ch, ratio=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // ratio), nn.ReLU(),
            nn.Linear(ch // ratio, ch), nn.Sigmoid())
    def forward(self, x):
        w = self.fc(x.mean(dim=[2,3])).unsqueeze(-1).unsqueeze(-1)
        return x * w

class CNNWithSE(nn.Module):
    def __init__(self, use_se=True):
        super().__init__()
        self.use_se = use_se
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2))
        self.se1 = SEBlock(64) if use_se else nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2))
        self.se2 = SEBlock(128) if use_se else nn.Identity()
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2))
        self.se3 = SEBlock(256) if use_se else nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(256*4*4, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, 10))
    def forward(self, x):
        x = self.se1(self.conv1(x))
        x = self.se2(self.conv2(x))
        x = self.se3(self.conv3(x))
        return self.fc(x.view(x.size(0), -1))

class DepthwiseSepConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))

class CNNWithDWConv(nn.Module):
    def __init__(self, use_dw=True):
        super().__init__()
        conv_fn = DepthwiseSepConv if use_dw else lambda i, o: nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
        self.features = nn.Sequential(
            conv_fn(3, 64), nn.MaxPool2d(2, 2),
            conv_fn(64, 128), nn.MaxPool2d(2, 2),
            conv_fn(128, 256), nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(
            nn.Linear(256*4*4, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, 10))
    def forward(self, x):
        return self.fc(self.features(x).view(x.size(0), -1))

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def train_and_eval(model, epochs=5, lr=0.001):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_losses, train_accs, test_accs = [], [], []
    for ep in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        scheduler.step()
        train_losses.append(running_loss / total)
        train_accs.append(correct / total)
        model.eval()
        tc, tt = 0, 0
        with torch.no_grad():
            for imgs, labels in testloader:
                out = model(imgs.to(device))
                tc += (out.argmax(1) == labels.to(device)).sum().item()
                tt += imgs.size(0)
        test_accs.append(tc / tt)
        log(f"    Epoch {ep+1}/{epochs}  train_loss={train_losses[-1]:.4f}  "
            f"train_acc={train_accs[-1]:.4f}  test_acc={test_accs[-1]:.4f}")
    return train_losses, train_accs, test_accs

# ============================================================
# 已有结果 (MLP和VGG)
# ============================================================
results_main = {
    'MLP': {
        'tl': [1.9012, 1.8088, 1.7461, 1.6987, 1.6465, 1.6088, 1.5739, 1.5494, 1.5250, 1.5125],
        'ta': [0.3117, 0.3580, 0.3795, 0.3939, 0.4099, 0.4205, 0.4331, 0.4420, 0.4510, 0.4568],
        'te': [0.3844, 0.4068, 0.4110, 0.4154, 0.4199, 0.4312, 0.4410, 0.4488, 0.4544, 0.4592],
        'time': 295.6, 'model': MLPClassifier()
    },
    'VGG-style': {
        'tl': [1.6249, 1.3670, 1.1727, 1.0142, 0.7313, 0.6876, 0.6003, 0.5455, 0.4892, 0.4399],
        'ta': [0.3974, 0.5064, 0.5835, 0.6414, 0.7506, 0.7601, 0.7902, 0.8100, 0.8338, 0.8537],
        'te': [0.5312, 0.6430, 0.7038, 0.7410, 0.7720, 0.7888, 0.8096, 0.8236, 0.8380, 0.8466],
        'time': 3447.6, 'model': VGGSmall()
    },
}

# ============================================================
# ResNet-18 训练 (5 epochs)
# ============================================================
log("\n" + "=" * 60)
log("[续训] ResNet-18 训练 (5 epochs)")
log("=" * 60)

log(f"\n--- 训练 ResNet-18 ---")
t0 = time.time()
resnet_model = ResNet18()
tl, ta, te = train_and_eval(resnet_model, epochs=5)
elapsed = time.time() - t0
results_main['ResNet-18'] = {'tl': tl, 'ta': ta, 'te': te, 'time': elapsed, 'model': resnet_model}
log(f"  耗时: {elapsed:.1f}s, 最终测试准确率: {te[-1]:.4f}")

# 训练曲线
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for name, r in results_main.items():
    axes[0].plot(r['tl'], label=name)
    axes[1].plot(r['ta'], label=name)
    axes[2].plot(r['te'], label=name)
for ax, title in zip(axes, ['训练损失', '训练准确率', '测试准确率']):
    ax.set_xlabel('Epoch'); ax.set_ylabel(title); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
save_fig(fig, 'main_comparison.png')

# 柱状图
names = list(results_main.keys())
# 补齐到相同长度
max_len = max(len(results_main[n]['te']) for n in names)
final_accs = []
for n in names:
    te = results_main[n]['te']
    final_accs.append(te[-1])
params = [count_params(results_main[n]['model']) for n in names]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(names, final_accs, color=['#e74c3c','#3498db','#2ecc71'])
axes[0].set_ylabel('测试准确率'); axes[0].set_title('最终测试准确率对比')
axes[0].set_ylim(0, 1)
for i, v in enumerate(final_accs):
    axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
axes[1].bar(names, params, color=['#e74c3c','#3498db','#2ecc71'])
axes[1].set_ylabel('参数量'); axes[1].set_title('模型参数量对比')
for i, v in enumerate(params):
    axes[1].text(i, v + max(params)*0.02, f'{v:,}', ha='center', fontsize=9)
save_fig(fig, 'main_bar_comparison.png')

# 预测样本
fig, axes = plt.subplots(3, 5, figsize=(14, 9))
for row, name in enumerate(names):
    model = results_main[name]['model']
    model.eval()
    for col in range(5):
        idx = col * 100
        img_arr = testset.data[idx]
        true_label = testset.targets[idx]
        img_t = transform_test(Image.fromarray(img_arr)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_t).argmax(1).item()
        axes[row, col].imshow(img_arr)
        color = 'green' if pred == true_label else 'red'
        axes[row, col].set_title(f'真:{CLASSES[true_label]}\n预:{CLASSES[pred]}',
                                 fontsize=8, color=color)
        axes[row, col].axis('off')
    axes[row, 0].set_ylabel(name, fontsize=11, rotation=90)
fig.suptitle('各模型预测结果示例 (绿色=正确, 红色=错误)', fontsize=13)
save_fig(fig, 'prediction_samples.png')

# ============================================================
# 注意力机制对比
# ============================================================
log("\n" + "=" * 60)
log("注意力机制对比: 有无SE模块")
log("=" * 60)

results_attn = {}
for name_key, use_se in [('CNN(无SE)', False), ('CNN(有SE)', True)]:
    log(f"\n--- 训练 {name_key} ---")
    m = CNNWithSE(use_se=use_se)
    tl, ta, te = train_and_eval(m, epochs=5)
    results_attn[name_key] = {'tl': tl, 'ta': ta, 'te': te, 'model': m}
    log(f"  最终测试准确率: {te[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for name, r in results_attn.items():
    axes[0].plot(r['tl'], label=name)
    axes[1].plot(r['te'], label=name)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('训练损失'); axes[0].set_title('SE注意力机制 — 训练损失')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('测试准确率'); axes[1].set_title('SE注意力机制 — 测试准确率')
for ax in axes: ax.legend(); ax.grid(True, alpha=0.3)
save_fig(fig, 'attention_comparison.png')

# ============================================================
# 卷积类型对比
# ============================================================
log("\n" + "=" * 60)
log("卷积类型对比: 标准卷积 vs 深度可分离卷积")
log("=" * 60)

results_conv = {}
for name_key, use_dw in [('CNN(标准卷积)', False), ('CNN(深度可分离)', True)]:
    log(f"\n--- 训练 {name_key} ---")
    m = CNNWithDWConv(use_dw=use_dw)
    tl, ta, te = train_and_eval(m, epochs=5)
    results_conv[name_key] = {'tl': tl, 'ta': ta, 'te': te, 'model': m}
    log(f"  最终测试准确率: {te[-1]:.4f}, 参数量: {count_params(m):,}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for name, r in results_conv.items():
    axes[0].plot(r['tl'], label=name)
    axes[1].plot(r['te'], label=name)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('训练损失'); axes[0].set_title('卷积类型 — 训练损失')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('测试准确率'); axes[1].set_title('卷积类型 — 测试准确率')
for ax in axes: ax.legend(); ax.grid(True, alpha=0.3)
save_fig(fig, 'conv_type_comparison.png')

# ============================================================
# 特征图可视化
# ============================================================
log("\n" + "=" * 60)
log("特征图可视化")
log("=" * 60)

vgg_model = results_main['VGG-style']['model']
vgg_model.eval()
sample_img = transform_test(Image.fromarray(testset.data[0])).unsqueeze(0).to(device)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes[0, 0].imshow(testset.data[0])
axes[0, 0].set_title('输入图像')
axes[0, 0].axis('off')

layer_names = ['Conv1(64)', 'Conv2(64)', 'Conv3(128)', 'Conv4(128)',
               'Conv5(256)', 'Conv6(256)']
feat_idx = 0
for i, layer in enumerate(vgg_model.features):
    if isinstance(layer, nn.ReLU) and feat_idx < 6:
        with torch.no_grad():
            partial = vgg_model.features[:i+1](sample_img)
        feat = partial[0, 0].cpu().numpy()
        row = 0 if feat_idx < 3 else 1
        col = (feat_idx % 3) + 1
        if row < 2 and col < 4:
            axes[row, col].imshow(feat, cmap='viridis')
            axes[row, col].set_title(f'{layer_names[feat_idx]}', fontsize=9)
            axes[row, col].axis('off')
        feat_idx += 1

for r in range(2):
    for c in range(4):
        if not axes[r, c].get_images():
            axes[r, c].axis('off')
fig.suptitle('VGG-style 特征图可视化', fontsize=14)
save_fig(fig, 'feature_maps.png')

# ============================================================
# 汇总
# ============================================================
log("\n" + "=" * 60)
log("实验汇总")
log("=" * 60)

log("\n--- 主对比实验结果 ---")
for name, r in results_main.items():
    log(f"  {name}: 测试准确率={r['te'][-1]:.4f}, 参数量={count_params(r['model']):,}, 耗时={r['time']:.1f}s")

log("\n--- SE注意力机制 ---")
for name, r in results_attn.items():
    log(f"  {name}: 测试准确率={r['te'][-1]:.4f}")

log("\n--- 卷积类型 ---")
for name, r in results_conv.items():
    log(f"  {name}: 测试准确率={r['te'][-1]:.4f}, 参数量={count_params(r['model']):,}")

log("\n实验完成!")
log_fh.close()
