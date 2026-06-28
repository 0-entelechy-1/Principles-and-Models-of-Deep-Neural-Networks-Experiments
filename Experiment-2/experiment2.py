import os, sys, time
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
log_fh = None

def log_open():
    global log_fh
    log_fh = open(LOG_FILE, 'w', encoding='utf-8')

def log(msg):
    print(msg)
    if log_fh:
        log_fh.write(str(msg) + '\n')
        log_fh.flush()

def save_fig(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name), bbox_inches='tight')
    plt.close(fig)
    log(f"  [saved] {name}")

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cpu')

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# ============================================================
# 1. 数据集
# ============================================================
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
# 2. 模型定义
# ============================================================

# --- MLP 基线 ---
class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3*32*32, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 10))
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

# --- VGG-style ---
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

# --- ResNet-18 ---
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

# --- SE Block ---
class SEBlock(nn.Module):
    def __init__(self, ch, ratio=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // ratio), nn.ReLU(),
            nn.Linear(ch // ratio, ch), nn.Sigmoid())
    def forward(self, x):
        w = self.fc(x.mean(dim=[2,3])).unsqueeze(-1).unsqueeze(-1)
        return x * w

# --- 带SE的简单CNN ---
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

# --- 深度可分离卷积 ---
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

# --- MobileNetV2 风格轻量模型 ---
class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio, use_se=False):
        super().__init__()
        assert stride in [1, 2]
        hidden = int(round(in_ch * expand_ratio))
        self.use_res = stride == 1 and in_ch == out_ch
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, 1, 0, bias=False),
                       nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.use_se = use_se
        self.se = SEBlock(out_ch, ratio=8) if use_se else None
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_se:
            out = self.se(out)
        if self.use_res:
            out = x + out
        return out

class MobileNetV2Like(nn.Module):
    """面向 CIFAR-10 的轻量 MobileNetV2 风格网络（缩减版，便于 CPU 训练）。"""
    def __init__(self, num_classes=10, use_se=False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            InvertedResidual(16, 16, 1, 1, use_se),
            InvertedResidual(16, 24, 2, 4, use_se),
            InvertedResidual(24, 24, 1, 4, use_se),
            InvertedResidual(24, 32, 2, 4, use_se),
            InvertedResidual(32, 32, 1, 4, use_se),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return self.classifier(x)

# --- ShuffleNetV2 风格轻量模型 ---
class ChannelShuffle(nn.Module):
    def __init__(self, groups=2):
        super().__init__()
        self.groups = groups
    def forward(self, x):
        n, c, h, w = x.size()
        g = self.groups
        return x.view(n, g, c // g, h, w).permute(0, 2, 1, 3, 4).contiguous().view(n, c, h, w)

class ShuffleV2Unit(nn.Module):
    def __init__(self, in_ch, out_ch, stride, use_shuffle=True):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        mid = out_ch // 2
        if stride == 2:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_ch, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
                nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False),
                nn.BatchNorm2d(mid),
                nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()
            self.branch2 = nn.Sequential(
                nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
                nn.Conv2d(mid, mid, 3, 1, 1, groups=mid, bias=False),
                nn.BatchNorm2d(mid),
                nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            )
        self.shuffle = ChannelShuffle(groups=2) if use_shuffle else nn.Identity()

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return self.shuffle(out)

class ShuffleNetV2Like(nn.Module):
    """面向 CIFAR-10 的轻量 ShuffleNetV2 风格网络（缩减版，便于 CPU 训练）。"""
    def __init__(self, num_classes=10, use_shuffle=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.stage2 = self._make_stage(16, 32, 2, 2, use_shuffle)
        self.stage3 = self._make_stage(32, 64, 2, 2, use_shuffle)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(128, num_classes)

    def _make_stage(self, in_ch, out_ch, stride, n_blocks, use_shuffle):
        layers = [ShuffleV2Unit(in_ch, out_ch, stride, use_shuffle)]
        for _ in range(1, n_blocks):
            layers.append(ShuffleV2Unit(out_ch, out_ch, 1, use_shuffle))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.conv5(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return self.fc(x)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

# ============================================================
# 3. 训练函数
# ============================================================
def train_and_eval(model, epochs=8, lr=0.001):
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
        if (ep+1) % 5 == 0 or ep == 0:
            log(f"    Epoch {ep+1}/{epochs}  train_loss={train_losses[-1]:.4f}  "
                f"train_acc={train_accs[-1]:.4f}  test_acc={test_accs[-1]:.4f}")
    return train_losses, train_accs, test_accs

def plot_curves(results, title_loss, title_acc, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, r in results.items():
        axes[0].plot(r['tl'], label=name)
        axes[1].plot(r['te'], label=name)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('训练损失'); axes[0].set_title(title_loss)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('测试准确率'); axes[1].set_title(title_acc)
    for ax in axes: ax.legend(); ax.grid(True, alpha=0.3)
    save_fig(fig, filename)

# ============================================================
# 4. 主实验流程
# ============================================================
def run_dataset_section():
    log("=" * 60)
    log("1. CIFAR-10 数据集加载")
    log("=" * 60)
    log(f"训练集: {len(trainset)}, 测试集: {len(testset)}")
    log(f"类别: {CLASSES}")

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for i, ax in enumerate(axes.flatten()):
        img_arr = trainset.data[i * 500]
        label = trainset.targets[i * 500]
        ax.imshow(img_arr)
        ax.set_title(f'{CLASSES[label]}', fontsize=9)
        ax.axis('off')
    fig.suptitle('CIFAR-10 数据集样本', fontsize=14)
    save_fig(fig, 'dataset_samples.png')

def run_model_summary(models):
    log("\n=== 模型参数量对比 ===")
    for name, m in models.items():
        log(f"  {name}: {count_params(m):,} 参数")

def run_main_comparison():
    log("\n" + "=" * 60)
    log("主对比实验: MLP vs VGG-style vs ResNet-18")
    log("=" * 60)
    EPOCHS_MAIN = 10
    results_main = {}
    for name in ['MLP', 'VGG-style', 'ResNet-18']:
        log(f"\n--- 训练 {name} ---")
        t0 = time.time()
        tl, ta, te = train_and_eval(MODELS[name], epochs=EPOCHS_MAIN)
        elapsed = time.time() - t0
        results_main[name] = {'tl': tl, 'ta': ta, 'te': te, 'time': elapsed}
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

    # 柱状图对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = list(results_main.keys())
    final_accs = [results_main[n]['te'][-1] for n in names]
    params = [count_params(MODELS[n]) for n in names]
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

    return results_main

def run_prediction_samples(results_main):
    fig, axes = plt.subplots(3, 5, figsize=(14, 9))
    names = list(results_main.keys())
    for row, name in enumerate(names):
        model = MODELS[name]
        model.eval()
        for col in range(5):
            idx = col * 100
            img_arr = testset.data[idx]
            true_label = testset.targets[idx]
            img_pil = Image.fromarray(img_arr)
            img_t = transform_test(img_pil).unsqueeze(0).to(device)
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

def run_attention_comparison():
    log("\n" + "=" * 60)
    log("注意力机制对比: 有无SE模块")
    log("=" * 60)
    results_attn = {}
    for name in ['CNN(无SE)', 'CNN(有SE)']:
        log(f"\n--- 训练 {name} ---")
        tl, ta, te = train_and_eval(MODELS[name], epochs=8)
        results_attn[name] = {'tl': tl, 'ta': ta, 'te': te}
        log(f"  最终测试准确率: {te[-1]:.4f}")
    plot_curves(results_attn, 'SE注意力机制 — 训练损失', 'SE注意力机制 — 测试准确率',
                'attention_comparison.png')
    return results_attn

def run_conv_comparison():
    log("\n" + "=" * 60)
    log("卷积类型对比: 标准卷积 vs 深度可分离卷积")
    log("=" * 60)
    results_conv = {}
    for name in ['CNN(标准卷积)', 'CNN(深度可分离)']:
        log(f"\n--- 训练 {name} ---")
        tl, ta, te = train_and_eval(MODELS[name], epochs=8)
        results_conv[name] = {'tl': tl, 'ta': ta, 'te': te}
        log(f"  最终测试准确率: {te[-1]:.4f}, 参数量: {count_params(MODELS[name]):,}")
    plot_curves(results_conv, '卷积类型 — 训练损失', '卷积类型 — 测试准确率',
                'conv_type_comparison.png')
    return results_conv

def run_mobile_shuff_comparison():
    """补充：在 MobileNet / ShuffleNet 风格模型上验证注意力与特殊卷积作用。"""
    log("\n" + "=" * 60)
    log("轻量模型验证: MobileNetV2 (注意力) 与 ShuffleNetV2 (特殊卷积)")
    log("=" * 60)

    log("\n--- MobileNetV2 注意力机制对比 ---")
    results_mobile = {}
    for name in ['MobileNetV2(无SE)', 'MobileNetV2(有SE)']:
        log(f"\n--- 训练 {name} ---")
        tl, ta, te = train_and_eval(MODELS[name], epochs=8)
        results_mobile[name] = {'tl': tl, 'ta': ta, 'te': te}
        log(f"  最终测试准确率: {te[-1]:.4f}, 参数量: {count_params(MODELS[name]):,}")
    plot_curves(results_mobile, 'MobileNetV2 注意力机制 — 训练损失',
                'MobileNetV2 注意力机制 — 测试准确率', 'mobilenet_attention_comparison.png')

    log("\n--- ShuffleNetV2 通道洗牌对比 ---")
    results_shuffle = {}
    for name in ['ShuffleNetV2(无shuffle)', 'ShuffleNetV2(有shuffle)']:
        log(f"\n--- 训练 {name} ---")
        tl, ta, te = train_and_eval(MODELS[name], epochs=8)
        results_shuffle[name] = {'tl': tl, 'ta': ta, 'te': te}
        log(f"  最终测试准确率: {te[-1]:.4f}, 参数量: {count_params(MODELS[name]):,}")
    plot_curves(results_shuffle, 'ShuffleNetV2 通道洗牌 — 训练损失',
                'ShuffleNetV2 通道洗牌 — 测试准确率', 'shufflenet_shuffle_comparison.png')

    # 综合柱状图
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results_mobile.keys()) + list(results_shuffle.keys())
    accs = [r['te'][-1] for r in list(results_mobile.values()) + list(results_shuffle.values())]
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
    bars = ax.bar(names, accs, color=colors)
    ax.set_ylabel('测试准确率'); ax.set_title('轻量模型消融实验最终准确率')
    ax.set_ylim(0, 1)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f'{v:.4f}',
                ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    save_fig(fig, 'mobile_shuffle_bar_comparison.png')

    return results_mobile, results_shuffle

def run_feature_maps():
    log("\n" + "=" * 60)
    log("特征图可视化")
    log("=" * 60)

    vgg_model = MODELS['VGG-style']
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
            if feat_idx < 4:
                row = 0; col = feat_idx + 1
            else:
                row = 1; col = feat_idx - 4 + 1
            if row < 2 and col < 4:
                axes[row, col].imshow(feat, cmap='viridis')
                axes[row, col].set_title(f'{layer_names[feat_idx]}', fontsize=9)
                axes[row, col].axis('off')
            feat_idx += 1

    for r in range(2):
        for c in range(4):
            axes[r, c].axis('off') if not axes[r, c].get_images() else None
    fig.suptitle('VGG-style 特征图可视化', fontsize=14)
    save_fig(fig, 'feature_maps.png')

def print_summary(results_main, results_attn, results_conv, results_mobile=None, results_shuffle=None):
    log("\n" + "=" * 60)
    log("实验汇总")
    log("=" * 60)

    log("\n--- 主对比实验结果 ---")
    for name, r in results_main.items():
        log(f"  {name}: 测试准确率={r['te'][-1]:.4f}, 参数量={count_params(MODELS[name]):,}, 耗时={r['time']:.1f}s")

    log("\n--- SE注意力机制 ---")
    for name, r in results_attn.items():
        log(f"  {name}: 测试准确率={r['te'][-1]:.4f}")

    log("\n--- 卷积类型 ---")
    for name, r in results_conv.items():
        log(f"  {name}: 测试准确率={r['te'][-1]:.4f}, 参数量={count_params(MODELS[name]):,}")

    if results_mobile:
        log("\n--- MobileNetV2 注意力机制 ---")
        for name, r in results_mobile.items():
            log(f"  {name}: 测试准确率={r['te'][-1]:.4f}, 参数量={count_params(MODELS[name]):,}")

    if results_shuffle:
        log("\n--- ShuffleNetV2 通道洗牌 ---")
        for name, r in results_shuffle.items():
            log(f"  {name}: 测试准确率={r['te'][-1]:.4f}, 参数量={count_params(MODELS[name]):,}")

    log("\n实验完成!")

MODELS = {
    'MLP': MLPClassifier(),
    'VGG-style': VGGSmall(),
    'ResNet-18': ResNet18(),
    'CNN(无SE)': CNNWithSE(use_se=False),
    'CNN(有SE)': CNNWithSE(use_se=True),
    'CNN(标准卷积)': CNNWithDWConv(use_dw=False),
    'CNN(深度可分离)': CNNWithDWConv(use_dw=True),
    'MobileNetV2(无SE)': MobileNetV2Like(use_se=False),
    'MobileNetV2(有SE)': MobileNetV2Like(use_se=True),
    'ShuffleNetV2(无shuffle)': ShuffleNetV2Like(use_shuffle=False),
    'ShuffleNetV2(有shuffle)': ShuffleNetV2Like(use_shuffle=True),
}

def main():
    log_open()
    run_dataset_section()
    run_model_summary(MODELS)
    results_main = run_main_comparison()
    run_prediction_samples(results_main)
    results_attn = run_attention_comparison()
    results_conv = run_conv_comparison()
    results_mobile, results_shuffle = run_mobile_shuff_comparison()
    run_feature_maps()
    print_summary(results_main, results_attn, results_conv, results_mobile, results_shuffle)
    if log_fh:
        log_fh.close()

if __name__ == '__main__':
    main()
