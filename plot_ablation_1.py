import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# 1. 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']

# 2. 自动寻找CSV文件的辅助函数
def get_csv_path(folder_path):
    files = glob.glob(f"{folder_path}/*.csv")
    if not files:
        print(f"⚠️ 警告：在 {folder_path} 找不到 CSV 文件！")
        return None
    return files[0]

# 3. 读取数据 (包含昨晚跑的 Baseline 和刚才跑的 4 个变体)
# 注意：Baseline 是 kernel=3, convdim=64
df_base = pd.read_csv('AntMaze/walker2d_lsdt.csv', on_bad_lines='skip') 

df_k5 = pd.read_csv(get_csv_path('AntMaze/ablation_k5'), on_bad_lines='skip')
df_k11 = pd.read_csv(get_csv_path('AntMaze/ablation_k11'), on_bad_lines='skip')

df_c32 = pd.read_csv(get_csv_path('AntMaze/ablation_c32'), on_bad_lines='skip')
df_c96 = pd.read_csv(get_csv_path('AntMaze/ablation_c96'), on_bad_lines='skip')

x_base = df_base['num_updates'] / 1000

# ==========================================
# 📊 图 1: Kernel Size 消融 (感受野)
# ==========================================
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(x_base, df_base['eval_d4rl_score'], label='Kernel=3', color='#e74c3c', linewidth=2.5)

if df_k5 is not None:
    plt.plot(df_k5['num_updates']/1000, df_k5['eval_d4rl_score'], label='Kernel=5', color='#3498db', linewidth=1.5, alpha=0.8)
if df_k11 is not None:
    plt.plot(df_k11['num_updates']/1000, df_k11['eval_d4rl_score'], label='Kernel=11', color='#95a5a6', linewidth=1.5, alpha=0.8)

plt.title('Ablation on Short-term Memory Receptive Field', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Training Updates (x1000 steps)', fontsize=14)
plt.ylabel('D4RL Normalized Score', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.xlim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('ablation_kernel.jpg')
plt.close() # 关闭当前画布，防止干扰下一张图
print("✅ 感受野消融图绘制完成！已保存为 ablation_kernel.jpg")


# ==========================================
# 📊 图 2: Conv Dimension 消融 (长短特征比)
# ==========================================
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(x_base, df_base['eval_d4rl_score'], label='ConvDim=64', color='#e74c3c', linewidth=2.5)

if df_c32 is not None:
    plt.plot(df_c32['num_updates']/1000, df_c32['eval_d4rl_score'], label='ConvDim=32 (Attn Dominant)', color='#2ecc71', linewidth=1.5, alpha=0.8)
if df_c96 is not None:
    plt.plot(df_c96['num_updates']/1000, df_c96['eval_d4rl_score'], label='ConvDim=96 (Conv Dominant)', color='#f39c12', linewidth=1.5, alpha=0.8)

plt.title('Ablation on Long-Short Feature Ratio', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Training Updates (x1000 steps)', fontsize=14)
plt.ylabel('D4RL Normalized Score', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.xlim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('ablation_convdim.jpg')
plt.close()
print("✅ 通道比消融图绘制完成！已保存为 ablation_convdim.jpg")