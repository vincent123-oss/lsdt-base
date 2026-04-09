import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
labels = ['Hopper-Medium', 'Walker2d-Medium', 'HalfCheetah-Medium']
# Baseline 数据来自原版 DT 论文
baseline_scores = [60.0, 72.0, 42.5]

lsdt_scores = [80.71, 79.16, 43.26]

x = np.arange(len(labels))  # 标签位置
width = 0.35  # 柱子宽度

# 2. 开始绘图 
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
rects1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline (DT Paper)', color='#E2A47D')
rects2 = ax.bar(x + width/2, lsdt_scores, width, label='Ours (LSDT)', color='#5C82B6')

# 3. 添加各种标签、标题和自定义 x 轴刻度标签
ax.set_ylabel('D4RL Normalized Score', fontsize=14, fontweight='bold')
ax.set_title('Performance Comparison on D4RL Benchmarks', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12, loc='upper left')

# 4. 添加网格线，让数据对比更清晰
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_axisbelow(True) 

# 5. 在柱子上自动标出具体数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移3个点
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# 6. 保存高质量图片
fig.tight_layout()
plt.savefig('final_benchmark_comparison.jpg', bbox_inches='tight')
print("✅ 柱状图绘制完成！已保存为 final_benchmark_comparison.jpg")