# make_test.py
code = r'''
import gym
import numpy as np
import torch
import d4rl
import sys
from decision_transformer.LSDT import DecisionTransformer

# ================= 配置区域 =================
MODEL_PATH = "AntMaze/0.0001_20_0.0001_3_5_model_25-12-23-17-17-20_best.pt" 
ENV_NAME = 'walker2d-medium-v2'
CONTEXT_LEN = 20
EMBED_DIM = 128
N_LAYER = 3
N_HEAD = 1
KERNEL_SIZE = 5      
CONV_DIM = 64        
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULT_FILE = "robustness_results.txt" 
# ===========================================

def log(msg):
    """同时打印到控制台和写入文件"""
    print(msg)
    with open(RESULT_FILE, "a") as f:
        f.write(msg + "\n")

def get_action_fn(model, states, actions, rewards, returns_to_go, timesteps):
    # 维度修正
    if states.dim() == 2: states = states.unsqueeze(0)
    if actions.dim() == 2: actions = actions.unsqueeze(0)
    if returns_to_go.dim() == 1: returns_to_go = returns_to_go.reshape(1, -1, 1)
    if timesteps.dim() == 1: timesteps = timesteps.reshape(1, -1)

    states = states[:, -CONTEXT_LEN:]
    actions = actions[:, -CONTEXT_LEN:]
    returns_to_go = returns_to_go[:, -CONTEXT_LEN:]
    timesteps = timesteps[:, -CONTEXT_LEN:]

    with torch.no_grad():
        # 按照 LSDT 定义顺序: (timesteps, states, actions, returns)
        _, action_preds, _ = model(
            timesteps.long(),   
            states,             
            actions,            
            returns_to_go       
        )

    return action_preds[0, -1]

def get_return(model, env, mean, std, noise_level=0.0):
    model.eval()
    model.to(DEVICE)
    state_mean = torch.from_numpy(mean).to(DEVICE)
    state_std = torch.from_numpy(std).to(DEVICE)

    # 50 轮评估 
    eval_episodes = 50
    total_scores = []
    
    log(f"Testing Noise Level: {noise_level} ...")
    
    for i in range(eval_episodes):
        state = env.reset()
        if noise_level > 0: state = state + np.random.normal(0, noise_level, state.shape)
            
        states = torch.from_numpy(state).reshape(1, 1, -1).to(device=DEVICE, dtype=torch.float32)
        actions = torch.zeros((1, 1, env.action_space.shape[0]), device=DEVICE, dtype=torch.float32)
        rewards = torch.zeros((1, 1, 1), device=DEVICE, dtype=torch.float32)
        returns_to_go = torch.tensor([1.0], device=DEVICE, dtype=torch.float32).reshape(1, 1, 1)
        timesteps = torch.tensor([0], device=DEVICE, dtype=torch.long).reshape(1, 1)

        episode_return = 0
        
        for t in range(1000):
            action = get_action_fn(model, (states - state_mean) / state_std, actions, rewards, returns_to_go, timesteps)
            action = action.detach().cpu().numpy()
            state, reward, done, _ = env.step(action)
            
            if noise_level > 0: state = state + np.random.normal(0, noise_level, state.shape)

            cur_state = torch.from_numpy(state).to(device=DEVICE).reshape(1, 1, -1)
            cur_action = torch.from_numpy(action).to(device=DEVICE).reshape(1, 1, -1)
            cur_reward = torch.tensor([reward], device=DEVICE).reshape(1, 1, 1)
            cur_rtg = (returns_to_go[0, -1, 0] - (reward/5000.0)).reshape(1, 1, 1)
            cur_timestep = torch.tensor([t + 1], device=DEVICE, dtype=torch.long).reshape(1, 1)

            actions = torch.cat([actions, cur_action], dim=1)
            returns_to_go = torch.cat([returns_to_go, cur_rtg], dim=1)
            states = torch.cat([states, cur_state], dim=1)
            timesteps = torch.cat([timesteps, cur_timestep], dim=1)

            episode_return += reward
            if done: break
        
        norm_score = env.get_normalized_score(episode_return) * 100
        total_scores.append(norm_score)
        
        # 每跑完10轮打印一个点，表示进度
        if (i + 1) % 10 == 0:
            print(f"  > Completed {i+1}/{eval_episodes} episodes...", end="\r")

    return np.mean(total_scores), np.std(total_scores)

if __name__ == '__main__':
    # 初始化清空文件
    with open(RESULT_FILE, "w") as f:
        f.write("Robustness Test Results (50 Episodes)\n")
        f.write("======================================\n")
        
    env = gym.make(ENV_NAME)
    dataset = d4rl.qlearning_dataset(env)
    state_mean = np.mean(dataset['observations'], axis=0)
    state_std = np.std(dataset['observations'], axis=0) + 1e-6
    
    model = DecisionTransformer(
        state_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        n_blocks=N_LAYER, h_dim=EMBED_DIM, context_len=CONTEXT_LEN,
        n_heads=N_HEAD, drop_p=0.1, kernelsize=KERNEL_SIZE, convdim=CONV_DIM, max_timestep=4096       
    )
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        
    log("\n🚀 Robustness Test (50 Episodes, Mean ± Std)")
    log(f"{'Noise':<10} | {'Score (Mean ± Std)':<25}")
    log("-" * 40)
    
    for noise in [0.0, 0.01, 0.05, 0.1]:
        m, s = get_return(model, env, state_mean, state_std, noise_level=noise)
        log(f"{noise:<10} | {m:.2f} ± {s:.2f}")
    
    print(f"\n✅ All results saved to {RESULT_FILE}")
'''

with open('test_robustness.py', 'w') as f:
    f.write(code)
    
print("✅ 脚本更新完毕：支持自动保存到 robustness_results.txt ！")