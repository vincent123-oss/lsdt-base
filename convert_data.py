import pickle
import numpy as np
import os

# 设置你的数据路径
INPUT_PATH = "./data/halfcheetah-medium-v2.pkl"
OUTPUT_PATH = "./data/halfcheetah-medium-v2_fixed.pkl"

def convert():
    print(f"正在加载: {INPUT_PATH}")
    with open(INPUT_PATH, 'rb') as f:
        data = pickle.load(f)

    # 检查是否已经是列表（如果是列表说明不需要转换）
    if isinstance(data, list):
        print("数据已经是一个列表了，无需转换。可能是代码读取路径有误。")
        return

    # 检查是否是扁平字典（包含 'observations' 等键）
    if isinstance(data, dict):
        print("检测到扁平字典格式，正在转换为轨迹列表...")
        
        # 获取所有键
        keys = list(data.keys())
        if 'observations' not in keys or 'terminals' not in keys:
            print("错误：数据缺少必要的 'observations' 或 'terminals' 键。")
            return

        N = data['observations'].shape[0]
        trajectories = []
        episode_start = 0

        # 遍历数据，根据 terminal 或 timeout 切分轨迹
        
        terminals = data['terminals']
        timeouts = data.get('timeouts', np.zeros_like(terminals))

        for i in range(N):
            done = terminals[i] or timeouts[i]
            
            # 如果到达回合结束，或者到达数据末尾
            if done or i == N - 1:
                # 切片创建单个轨迹
                episode_data = {}
                for k in keys:
                    if isinstance(data[k], np.ndarray) and len(data[k]) == N:
                        episode_data[k] = data[k][episode_start : i + 1]
                    else:
                        # 某些元数据直接复制
                        episode_data[k] = data[k]
                
                trajectories.append(episode_data)
                episode_start = i + 1

        print(f"转换完成！共生成 {len(trajectories)} 条轨迹。")
        
        # 保存新文件
        with open(OUTPUT_PATH, 'wb') as f:
            pickle.dump(trajectories, f)
        print(f"新文件已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    convert()
