"""
实验二补充脚本：单独训练 MobileNetV2 与 ShuffleNetV2 风格模型，
验证注意力机制（SE）与特殊卷积（Channel Shuffle / 深度可分离）的作用。
避免重复运行已有的大量基线实验，节省 CPU 训练时间。
"""
import os, sys, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 从主实验脚本导入公共定义（模型、数据、训练函数等）
from experiment2 import (
    OUTPUT_DIR, CLASSES, device, log as _base_log, save_fig,
    train_and_eval, count_params, MODELS
)

SUPPLEMENT_LOG = os.path.join(OUTPUT_DIR, 'experiment2_supplement_log.txt')
_sup_fh = open(SUPPLEMENT_LOG, 'w', encoding='utf-8')

def log(msg):
    print(msg)
    _sup_fh.write(str(msg) + '\n')
    _sup_fh.flush()

# 覆盖 save_fig 内部使用的 log，使其也写入补充日志
def save_fig_local(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name), bbox_inches='tight')
    plt.close(fig)
    log(f"  [saved] {name}")


def plot_curves(results, title_loss, title_acc, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, r in results.items():
        axes[0].plot(r['tl'], label=name)
        axes[1].plot(r['te'], label=name)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('训练损失'); axes[0].set_title(title_loss)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('测试准确率'); axes[1].set_title(title_acc)
    for ax in axes: ax.legend(); ax.grid(True, alpha=0.3)
    save_fig_local(fig, filename)


def main():
    log("=" * 60)
    log("实验二补充：MobileNetV2 / ShuffleNetV2 消融实验")
    log("=" * 60)

    log("\n=== 新增模型参数量 ===")
    for name in ['MobileNetV2(无SE)', 'MobileNetV2(有SE)',
                 'ShuffleNetV2(无shuffle)', 'ShuffleNetV2(有shuffle)']:
        log(f"  {name}: {count_params(MODELS[name]):,} 参数")

    log("\n--- MobileNetV2 注意力机制对比 ---")
    results_mobile = {}
    for name in ['MobileNetV2(无SE)', 'MobileNetV2(有SE)']:
        log(f"\n--- 训练 {name} ---")
        t0 = time.time()
        tl, ta, te = train_and_eval(MODELS[name], epochs=5)
        elapsed = time.time() - t0
        results_mobile[name] = {'tl': tl, 'ta': ta, 'te': te, 'time': elapsed}
        log(f"  耗时: {elapsed:.1f}s, 最终测试准确率: {te[-1]:.4f}, "
            f"参数量: {count_params(MODELS[name]):,}")
    plot_curves(results_mobile,
                'MobileNetV2 注意力机制 — 训练损失',
                'MobileNetV2 注意力机制 — 测试准确率',
                'mobilenet_attention_comparison.png')

    log("\n--- ShuffleNetV2 通道洗牌对比 ---")
    results_shuffle = {}
    for name in ['ShuffleNetV2(无shuffle)', 'ShuffleNetV2(有shuffle)']:
        log(f"\n--- 训练 {name} ---")
        t0 = time.time()
        tl, ta, te = train_and_eval(MODELS[name], epochs=5)
        elapsed = time.time() - t0
        results_shuffle[name] = {'tl': tl, 'ta': ta, 'te': te, 'time': elapsed}
        log(f"  耗时: {elapsed:.1f}s, 最终测试准确率: {te[-1]:.4f}, "
            f"参数量: {count_params(MODELS[name]):,}")
    plot_curves(results_shuffle,
                'ShuffleNetV2 通道洗牌 — 训练损失',
                'ShuffleNetV2 通道洗牌 — 测试准确率',
                'shufflenet_shuffle_comparison.png')

    # 综合柱状图
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results_mobile.keys()) + list(results_shuffle.keys())
    accs = [r['te'][-1] for r in list(results_mobile.values()) + list(results_shuffle.values())]
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
    bars = ax.bar(names, accs, color=colors)
    ax.set_ylabel('测试准确率')
    ax.set_title('轻量模型消融实验最终准确率')
    ax.set_ylim(0, 1)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f'{v:.4f}',
                ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    save_fig_local(fig, 'mobile_shuffle_bar_comparison.png')

    log("\n" + "=" * 60)
    log("补充实验汇总")
    log("=" * 60)
    log("\n--- MobileNetV2 注意力机制 ---")
    for name, r in results_mobile.items():
        log(f"  {name}: 测试准确率={r['te'][-1]:.4f}, "
            f"参数量={count_params(MODELS[name]):,}, 耗时={r['time']:.1f}s")
    log("\n--- ShuffleNetV2 通道洗牌 ---")
    for name, r in results_shuffle.items():
        log(f"  {name}: 测试准确率={r['te'][-1]:.4f}, "
            f"参数量={count_params(MODELS[name]):,}, 耗时={r['time']:.1f}s")
    log("\n补充实验完成!")
    _sup_fh.close()


if __name__ == '__main__':
    main()
