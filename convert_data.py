import gym
import d4rl # 必须导入，触发注册机制
import pickle
import numpy as np
import os

# 你想要自动下载和转换的环境列表 (你可以随时在这里添加 hopper, walker2d)
ENV_NAMES = [
    'halfcheetah-medium-v2',
    'walker2d-medium-v2',
    'hopper-medium-v2'
]

OUTPUT_DIR = "./data"

def download_and_convert(env_name):
    print(f"\n=============================================")
    print(f"🚀 开始处理环境: {env_name}")
    print(f"=============================================")
    
    # 1. 自动创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{env_name}.pkl")
    
    # 2. 触发 D4RL 自动下载机制！
    print(f"正在通过 D4RL 拉取数据 (如果是首次运行，会自动从云端下载，请保持网络畅通)...")
    env = gym.make(env_name)
    data = env.get_dataset() # 魔法就在这一行！
    
    # 3. 验证数据格式
    keys = list(data.keys())
    if 'observations' not in keys or 'terminals' not in keys:
        print("❌ 错误：拉取的数据缺少必要的键。")
        return

    print("✅ 数据拉取成功！开始切分为轨迹序列...")
    
    # 4. 开始切分轨迹 (你的核心逻辑)
    N = data['observations'].shape[0]
    trajectories = []
    episode_start = 0
    
    terminals = data['terminals']
    timeouts = data.get('timeouts', np.zeros_like(terminals))

    for i in range(N):
        done = terminals[i] or timeouts[i]
        if done or i == N - 1:
            episode_data = {}
            for k in keys:
                if isinstance(data[k], np.ndarray) and len(data[k]) == N:
                    episode_data[k] = data[k][episode_start : i + 1]
                else:
                    episode_data[k] = data[k]
            
            trajectories.append(episode_data)
            episode_start = i + 1

    print(f"🔄 转换完成！共生成 {len(trajectories)} 条轨迹。")
    
    # 5. 保存到本地 data/ 文件夹
    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"💾 最终数据已保存至: {output_path}")

if __name__ == "__main__":
    # 批量跑完三大环境
    for env in ENV_NAMES:
        try:
            download_and_convert(env)
        except Exception as e:
            print(f"❌ 处理 {env} 时发生错误: {e}")
            print("可能是由于国内网络无法连接 D4RL 服务器，请尝试配置科学上网环境。")