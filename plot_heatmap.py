import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from decision_transformer.LSDT import DecisionTransformer

print("Loading model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 实例化模型
model = DecisionTransformer(
    state_dim=17, act_dim=6, n_blocks=3,
    h_dim=128, context_len=20, n_heads=2,
    drop_p=0.1, kernelsize=3, convdim=64
).to(device)

# 2. 挂载最佳权重
model.load_state_dict(torch.load('AntMaze/walker2d_lsdt_best.pt', map_location=device))
model.eval()

# 3. 一段历史输入
B, T = 1, 20
timesteps = torch.arange(T, device=device).unsqueeze(0)
states = torch.randn(B, T, 17, device=device)
actions = torch.randn(B, T, 6, device=device)
returns_to_go = torch.randn(B, T, 1, device=device)

with torch.no_grad():
    model(timesteps, states, actions, returns_to_go)

# 4. 获取注意力矩阵 (真实大小为 60x60)
attn_matrix = model.transformer[0].attention.saved_attn_weights[0, 0].cpu().numpy()

# 5. 画图优化：解决坐标重叠问题
plt.figure(figsize=(10, 8), dpi=300)

# 每隔 3 个格子（RTG, State, Action）打一个刻度，定位在中间 (1.5, 4.5, 7.5...)
tick_positions = np.arange(1.5, 60, 3) 
tick_labels = [f"t={i}" for i in range(1, 21)]

ax = sns.heatmap(attn_matrix, cmap='viridis', 
                 cbar_kws={'label': 'Attention Weight'})

# 重新设置 xy 轴的刻度与倾斜度
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
ax.set_yticks(tick_positions)
ax.set_yticklabels(tick_labels, rotation=0, fontsize=10)

plt.title("LSDT Self-Attention Weights (Walker2d, Layer 1, Head 1)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Key Tokens (RTG, State, Action sequence)", fontsize=14)
plt.ylabel("Query Tokens", fontsize=14)

plt.tight_layout()
plt.savefig('attention_heatmap.jpg')
print("✅ 热力图绘制完成！坐标轴已彻底修复！")