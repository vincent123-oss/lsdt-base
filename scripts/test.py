import argparse
import os
import sys
import gym
import torch
import numpy as np
import d4rl
import random

from decision_transformer.utils_o import evaluate_on_env, get_d4rl_normalized_score, get_d4rl_dataset_stats
from decision_transformer.LSDT import DecisionTransformer

# 1. 增加抗扰动包装器，用于测试环境中的状态偏移和输入噪声
class RobustEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_scale=0.0, state_shift=0.0):
        super().__init__(env)
        self.noise_scale = noise_scale
        self.state_shift = state_shift
        
    def observation(self, obs):
        # 注入高斯噪声和固定的状态偏移
        obs = obs + np.random.normal(0, self.noise_scale, size=obs.shape)
        obs = obs + self.state_shift
        return obs

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def test(args):
    # 2. 统一训练和测试的 Target RTG，修复硬编码
    eval_dataset = args.dataset
    eval_rtg_scale = args.rtg_scale
    kernelsize = args.kernel_size

    if args.env == 'walker2d':
        eval_env_name = 'Walker2d-v2'
        eval_rtg_target = 5000  # 与 train.py 保持一致
        eval_env_d4rl_name = f'walker2d-{eval_dataset}-v2'
    elif args.env == 'halfcheetah':
        eval_env_name = 'HalfCheetah-v2'
        eval_rtg_target = 6000  # 与 train.py 保持一致
        eval_env_d4rl_name = f'halfcheetah-{eval_dataset}-v2'
    elif args.env == 'hopper':
        eval_env_name = 'Hopper-v2'  # 修复：去除了 'hopper-medium-replay' 的硬编码
        eval_rtg_target = 3600       # 与 train.py 保持一致
        eval_env_d4rl_name = f'hopper-{eval_dataset}-v2'
    elif args.env == 'maze2d':
        eval_env_name = f'maze2d-{eval_dataset}-v1'
        eval_rtg_target = 300
        eval_env_d4rl_name = f'maze2d-{eval_dataset}-v1' # 修复：之前错误写成了 hopper
    elif args.env == 'antmaze':
        eval_env_name = f'antmaze-{eval_dataset}-v2'
        eval_rtg_target = 100
        eval_env_d4rl_name = f'antmaze-{eval_dataset}-v2'
    else:
        raise NotImplementedError

    device = torch.device(args.device)
    
    env_data_stats = get_d4rl_dataset_stats(eval_env_d4rl_name)
    eval_state_mean = np.array(env_data_stats['state_mean'])
    eval_state_std = np.array(env_data_stats['state_std'])

    # 3. 多随机种子循环评估
    for seed in args.seeds:
        set_seeds(seed)
        print(f"\n{'='*20} Evaluating Seed: {seed} {'='*20}")
        
        raw_env = gym.make(eval_env_name)
        raw_env.seed(seed)
        
        # 应用抗扰动 Wrapper
        eval_env = RobustEnvWrapper(raw_env, noise_scale=args.noise_scale, state_shift=args.state_shift)
        
        state_dim = eval_env.observation_space.shape[0] + 2 if args.goalconcate else eval_env.observation_space.shape[0]
        act_dim = eval_env.action_space.shape[0]

        eval_model = DecisionTransformer(
            state_dim=state_dim, act_dim=act_dim, n_blocks=args.n_blocks,
            h_dim=args.embed_dim, context_len=args.context_len, n_heads=args.n_heads,
            drop_p=args.dropout_p, kernelsize=kernelsize, convdim=args.convdim,
        ).to(device)

        eval_chk_pt_path = os.path.join(args.chk_pt_dir, args.chk_pt_name)
        eval_model.load_state_dict(torch.load(eval_chk_pt_path, map_location=device))

        results = evaluate_on_env(eval_model, device, args.context_len,
                                eval_env, eval_rtg_target, eval_rtg_scale,
                                args.num_eval_ep, args.max_eval_ep_len, args.goalconcate,
                                eval_state_mean, eval_state_std, render=args.render)
                                
        norm_score = get_d4rl_normalized_score(results['eval/avg_reward'], eval_env_name) * 100
        print(f"Seed {seed} - Normalized D4RL Score: {norm_score:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 增加多种子与抗扰动参数
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456], help='List of seeds for evaluation')
    parser.add_argument('--noise_scale', type=float, default=0.0, help='Std dev of Gaussian noise for states')
    parser.add_argument('--state_shift', type=float, default=0.0, help='Constant bias added to states')
    # ... 保留原有其他参数 ...