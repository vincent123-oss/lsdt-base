#!/bin/bash
set -e

echo "=================================================="
echo ">>> 开始自动配置 LSDT 项目环境 (mujoco_py37)"
echo "=================================================="

echo ">>> 1. 创建并激活 conda 虚拟环境"
# 使用命令行直接创建，避免 environment.yml 里的名字冲突，加上 || true 防止已存在时报错中断
conda create -n mujoco_py37 python=3.7 -y || true

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mujoco_py37

echo ">>> 2. 安装 mujoco-py 所需的系统级底层依赖"
sudo apt update
sudo apt install -y patchelf libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev libegl1-mesa

echo ">>> 3. 下载并解压 Mujoco 2.1.0 物理引擎"
mkdir -p ~/.mujoco
cd ~/.mujoco
# 检查是否已经下载过
if [ ! -d "mujoco210" ]; then
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz
    tar -xzf mujoco210.tar.gz
    rm mujoco210.tar.gz
else
    echo "Mujoco 2.1.0 已存在，跳过下载。"
fi

echo ">>> 4. 配置系统环境变量 (~/.bashrc)"
# 使用 grep 检查是否已经写入过
if ! grep -q "mujoco210/bin" ~/.bashrc; then
    echo '' >> ~/.bashrc
    echo '# MuJoCo Paths for LSDT Project' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
    echo 'export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210' >> ~/.bashrc
    echo 'export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mjkey.txt' >> ~/.bashrc
    echo "环境变量已写入 ~/.bashrc"
else
    echo "环境变量已存在，无需重复写入。"
fi

echo ">>> 生效环境变量..."
# 注意：在 bash 脚本中 source ~/.bashrc 仅对当前执行的子 shell 有效
# 跑完脚本后手动 source 一次
eval "$(cat ~/.bashrc | tail -n 10)"

echo ">>> 5. 回到项目"
cd ~/Long-Short_Decision_Transformer/lsdt\ base || cd ~/Long-Short_Decision_Transformer

echo ">>> 6. 安装核心 Python 依赖包"
echo ">>> 安装 mujoco-py（首次安装会触发 gcc 编译，请耐心等待几分钟）..."
pip install 'mujoco-py<2.2,>=2.1'

echo ">>> 安装 D4RL 最新版..."
pip install git+https://github.com/Farama-Foundation/d4rl@master

echo ">>> 安装 Gym (强依赖 0.21 版本)..."
pip install gym==0.21

echo ">>> 安装其他要求依赖 (从 requirements.txt)..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "未找到 requirements.txt，跳过补充依赖安装。"
fi

echo "=================================================="
echo "LSDT 环境配置全部完成！"
echo "=================================================="
echo "在当前终端复制并执行以下两行命令来彻底激活环境："
echo "source ~/.bashrc"
echo "conda activate mujoco_py37"
echo "=================================================="