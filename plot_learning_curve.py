import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']

# 2. 读取 CSV 文件
df_walker = pd.read_csv('AntMaze/walker2d_lsdt.csv', on_bad_lines='skip')
df_hopper = pd.read_csv('AntMaze/hopper_lsdt.csv', on_bad_lines='skip')
df_cheetah = pd.read_csv('AntMaze/halfcheetah_lsdt.csv', on_bad_lines='skip')

# 3. 创建画布
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 提取更新步数 (x轴)
x_walker = df_walker['num_updates'] / 1000
x_hopper = df_hopper['num_updates'] / 1000
x_cheetah = df_cheetah['num_updates'] / 1000

# 4. 绘制完全真实的原始数据曲线 (Raw Data)
ax.plot(x_walker, df_walker['eval_d4rl_score'], 
        label='Walker2d-Medium (Peak: 79.16)', color='#3498db', linewidth=2.0)
ax.plot(x_hopper, df_hopper['eval_d4rl_score'], 
        label='Hopper-Medium (Peak: 80.71)', color='#e74c3c', linewidth=2.0)
ax.plot(x_cheetah, df_cheetah['eval_d4rl_score'], 
        label='HalfCheetah-Medium (Peak: 43.26)', color='#2ecc71', linewidth=2.0)

# 5. 美化图表细节
ax.set_title('Training Convergence Analysis of LSDT (Raw Data)', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Training Updates (x1000 steps)', fontsize=14)
ax.set_ylabel('D4RL Normalized Score', fontsize=14)
ax.legend(fontsize=12, loc='lower right')
ax.set_xlim(0, 100) # x轴限制在 0-100k 步

# 增加网格线对比度
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('final_learning_curve_raw.jpg')
print(" 训练学习曲线绘制完成！已保存为 final_learning_curve_raw.jpg")