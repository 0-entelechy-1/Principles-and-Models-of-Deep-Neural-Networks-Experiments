import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, 'experiment_log.txt')
log_handle = open(LOG_FILE, 'w', encoding='utf-8')


def log(msg):
    print(msg)
    log_handle.write(str(msg) + '\n')
    log_handle.flush()


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    log(f"  [saved] {name}")


torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cpu')
log(f"设备: {device}")
log(f"PyTorch: {torch.__version__}")

# ============================================================
# 1. 数据集构建
# ============================================================
log("\n" + "=" * 60)
log("1. 数据集构建")
log("=" * 60)

N_SAMPLES = 300
NOISE_STD = 0.1

x_all = np.linspace(-np.pi, np.pi, N_SAMPLES).reshape(-1, 1).astype(np.float32)
y_all = (np.sin(x_all) + 0.5 * np.cos(2 * x_all)
         + np.random.normal(0, NOISE_STD, x_all.shape).astype(np.float32))

x_tensor = torch.from_numpy(x_all)
y_tensor = torch.from_numpy(y_all)

perm = np.random.permutation(N_SAMPLES)
n_train, n_val = int(0.7 * N_SAMPLES), int(0.15 * N_SAMPLES)

train_idx, val_idx, test_idx = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]

x_train, y_train = x_tensor[train_idx], y_tensor[train_idx]
x_val, y_val = x_tensor[val_idx], y_tensor[val_idx]
x_test, y_test = x_tensor[test_idx], y_tensor[test_idx]

log(f"训练集: {len(x_train)}, 验证集: {len(x_val)}, 测试集: {len(x_test)}")

x_smooth_np = np.linspace(-np.pi, np.pi, 500).reshape(-1, 1).astype(np.float32)
y_smooth_np = np.sin(x_smooth_np) + 0.5 * np.cos(2 * x_smooth_np)
x_smooth_t = torch.from_numpy(x_smooth_np)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x_all, y_all, s=5, alpha=0.5, label='带噪声数据')
ax.plot(x_smooth_np, y_smooth_np, 'r-', lw=2, label='真实函数: sin(x)+0.5cos(2x)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('数据集分布')
ax.legend()
ax.grid(True, alpha=0.3)
save_fig(fig, 'dataset.png')


# ============================================================
# 2. MLP 模型
# ============================================================
class MLP(nn.Module):
    def __init__(self, hidden_sizes, activation='relu'):
        super().__init__()
        act_map = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        layers = []
        in_size = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(act_map[activation]())
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


MODEL_CONFIGS = {
    '浅层(1x64,ReLU)':   {'hidden_sizes': [64],               'activation': 'relu'},
    '中层(2x64,ReLU)':   {'hidden_sizes': [64, 64],           'activation': 'relu'},
    '深层(4x64,ReLU)':   {'hidden_sizes': [64, 64, 64, 64],   'activation': 'relu'},
    '深层(4x64,Sigmoid)': {'hidden_sizes': [64, 64, 64, 64],  'activation': 'sigmoid'},
}

log("\n=== 模型结构 ===")
for name, cfg in MODEL_CONFIGS.items():
    m = MLP(**cfg)
    log(f"  {name}: 参数量={count_params(m)}")


# ============================================================
# 3. 训练函数
# ============================================================
def train_model(model, opt_cls, lr=0.01, epochs=500, batch_size=32, **opt_kw):
    criterion = nn.MSELoss()
    optimizer = opt_cls(model.parameters(), lr=lr, **opt_kw)
    train_losses, val_losses = [], []
    n_params = len(list(model.parameters()))
    grad_norms = {i: [] for i in range(n_params)}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(x_train.size(0))
        epoch_loss, n_batch = 0.0, 0

        for i in range(0, x_train.size(0), batch_size):
            idx = perm[i:i + batch_size]
            bx, by = x_train[idx], y_train[idx]
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()

            for li, p in enumerate(model.parameters()):
                if p.grad is not None and li < n_params:
                    grad_norms[li].append(p.grad.data.norm(2).item())

            optimizer.step()
            epoch_loss += loss.item()
            n_batch += 1

        train_losses.append(epoch_loss / n_batch)

        model.eval()
        with torch.no_grad():
            val_losses.append(criterion(model(x_val), y_val).item())

        if (epoch + 1) % 100 == 0:
            log(f"    Epoch {epoch+1}/{epochs}  train={train_losses[-1]:.6f}  val={val_losses[-1]:.6f}")

    return train_losses, val_losses, grad_norms


# ============================================================
# 4. 对比实验1: 优化器
# ============================================================
log("\n" + "=" * 60)
log("对比实验1: 优化器对比 (SGD / SGD+Momentum / Adam)")
log("=" * 60)

OPT_CFGS = {
    'SGD':          (optim.SGD, {'lr': 0.01}),
    'SGD+Momentum': (optim.SGD, {'lr': 0.01, 'momentum': 0.9}),
    'Adam':         (optim.Adam, {'lr': 0.01}),
}

res_opt = {}
for name, (cls, kw) in OPT_CFGS.items():
    log(f"\n--- {name} ---")
    m = MLP(**MODEL_CONFIGS['中层(2x64,ReLU)'])
    tl, vl, gn = train_model(m, cls, epochs=500, batch_size=32, **kw)
    res_opt[name] = {'model': m, 'tl': tl, 'vl': vl, 'gn': gn}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, r in res_opt.items():
    axes[0].plot(r['tl'], label=name)
    axes[1].plot(r['vl'], label=name)
for ax, title in zip(axes, ['训练损失', '验证损失']):
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'不同优化器 — {title}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
save_fig(fig, 'optimizer_comparison.png')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, r) in enumerate(res_opt.items()):
    r['model'].eval()
    with torch.no_grad():
        yp = r['model'](x_smooth_t).numpy()
    axes[idx].scatter(x_all, y_all, s=3, alpha=0.3, label='数据')
    axes[idx].plot(x_smooth_np, y_smooth_np, 'r-', lw=2, label='真实')
    axes[idx].plot(x_smooth_np, yp, 'g--', lw=2, label=f'{name}')
    axes[idx].set_title(f'{name} 拟合')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)
save_fig(fig, 'optimizer_fitting.png')


# ============================================================
# 5. 对比实验2: 网络结构 + 激活函数
# ============================================================
log("\n" + "=" * 60)
log("对比实验2: 网络结构与激活函数对比")
log("=" * 60)

res_struct = {}
for name, cfg in MODEL_CONFIGS.items():
    log(f"\n--- {name} ---")
    m = MLP(**cfg)
    tl, vl, gn = train_model(m, optim.Adam, lr=0.01, epochs=500, batch_size=32)
    m.eval()
    with torch.no_grad():
        test_mse = nn.MSELoss()(m(x_test), y_test).item()
    res_struct[name] = {'model': m, 'tl': tl, 'vl': vl, 'gn': gn, 'test_mse': test_mse}
    log(f"  测试MSE = {test_mse:.6f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, r in res_struct.items():
    axes[0].plot(r['tl'], label=name)
    axes[1].plot(r['vl'], label=name)
for ax, title in zip(axes, ['训练损失', '验证损失']):
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'不同网络结构 — {title}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
save_fig(fig, 'structure_comparison.png')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for idx, (name, r) in enumerate(res_struct.items()):
    r['model'].eval()
    with torch.no_grad():
        yp = r['model'](x_smooth_t).numpy()
    axes[idx].scatter(x_all, y_all, s=3, alpha=0.3, label='数据')
    axes[idx].plot(x_smooth_np, y_smooth_np, 'r-', lw=2, label='真实')
    axes[idx].plot(x_smooth_np, yp, 'g--', lw=2, label='拟合')
    axes[idx].set_title(f'{name} (MSE={r["test_mse"]:.6f})')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)
save_fig(fig, 'structure_fitting.png')


# ============================================================
# 6. 对比实验3: 学习率
# ============================================================
log("\n" + "=" * 60)
log("对比实验3: 学习率对比")
log("=" * 60)

LR_LIST = [0.001, 0.01, 0.1, 1.0]
res_lr = {}
for lr in LR_LIST:
    log(f"\n--- lr={lr} ---")
    m = MLP(**MODEL_CONFIGS['中层(2x64,ReLU)'])
    tl, vl, gn = train_model(m, optim.SGD, lr=lr, epochs=300, batch_size=32)
    res_lr[f'lr={lr}'] = {'model': m, 'tl': tl, 'vl': vl}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, r in res_lr.items():
    axes[0].plot(r['tl'], label=name)
    axes[1].plot(r['vl'], label=name)
for ax, title in zip(axes, ['训练损失', '验证损失']):
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'不同学习率 (SGD) — {title}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
save_fig(fig, 'lr_comparison.png')


# ============================================================
# 7. 梯度消失/爆炸分析
# ============================================================
log("\n" + "=" * 60)
log("梯度消失与梯度爆炸分析")
log("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for idx, (name, r) in enumerate(res_struct.items()):
    ax = axes[idx]
    for li, norms in r['gn'].items():
        if norms:
            step = max(1, len(norms) // 500)
            ax.plot(norms[::step], label=f'层{li}', alpha=0.7)
    ax.set_xlabel('训练步数')
    ax.set_ylabel('梯度L2范数')
    ax.set_title(f'{name}')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
save_fig(fig, 'gradient_analysis.png')

# 6层 Sigmoid 详细分析
log("\n--- 6层Sigmoid网络梯度消失详细分析 ---")
deep_sig = MLP(hidden_sizes=[64]*6, activation='sigmoid')
log(f"  参数量: {count_params(deep_sig)}")

criterion = nn.MSELoss()
opt_ds = optim.Adam(deep_sig.parameters(), lr=0.01)
n_p = len(list(deep_sig.parameters()))
layer_grads = {i: [] for i in range(n_p)}

for ep in range(200):
    deep_sig.train()
    opt_ds.zero_grad()
    loss = criterion(deep_sig(x_train), y_train)
    loss.backward()
    for li, p in enumerate(deep_sig.parameters()):
        if p.grad is not None and li < n_p:
            layer_grads[li].append(p.grad.data.norm(2).item())
    opt_ds.step()

fig, ax = plt.subplots(figsize=(10, 6))
for li, norms in layer_grads.items():
    if li % 2 == 0:
        ax.plot(norms, label=f'第{li//2+1}层权重', alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('梯度L2范数')
ax.set_title('6层Sigmoid网络 — 梯度消失现象')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
save_fig(fig, 'gradient_vanishing.png')


# ============================================================
# 8. 最佳模型评估
# ============================================================
log("\n" + "=" * 60)
log("最佳模型评估")
log("=" * 60)

best_name = min(res_struct, key=lambda k: res_struct[k]['test_mse'])
best_m = res_struct[best_name]['model']
best_mse = res_struct[best_name]['test_mse']
log(f"最佳: {best_name}, 测试MSE={best_mse:.6f}")

best_m.eval()
with torch.no_grad():
    yp = best_m(x_smooth_t).numpy()

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_all, y_all, s=5, alpha=0.3, label='数据')
ax.plot(x_smooth_np, y_smooth_np, 'r-', lw=2, label='真实函数')
ax.plot(x_smooth_np, yp, 'g--', lw=2, label=f'最佳拟合 ({best_name})')
ax.set_title(f'最佳模型拟合 (测试MSE={best_mse:.6f})')
ax.legend()
ax.grid(True, alpha=0.3)
save_fig(fig, 'best_model_fitting.png')

log("\n" + "=" * 60)
log("实验完成!")
log("=" * 60)
log_handle.close()
