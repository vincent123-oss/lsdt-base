
Long-Short Decision Transformer (LSDT)

**用于离线强化学习的长短特征融合决策大模型**

本项目提出了一种基于 **前置归一化 (Pre-LN)** 和 **Sigmoid 门控机制** 的长短特征融合决策架构。在 D4RL 离线强化学习基准测试中，LSDT 全越了原版的 Decision Transformer，并在各种复杂连续控制任务 (如 Walker2d, Hopper, HalfCheetah) 中展现出极高的训练稳定性和峰值性能。

-----


## 1\. 系统与硬件要求

  * 操作系统: 推荐 Linux 22.04或 WSL2。
  * GPU: 极度轻量化。由于模型架构优化，单任务训练仅需占用约 **650MB VRAM**，市面主流显卡甚至轻薄本即可流畅运行。
  * 依赖核心: Python 3.7, PyTorch, MuJoCo 2.1.0, D4RL。

-----

## 2\. 环境配置 (Environment Setup)

本项目涉及到底层物理仿真引擎，手动配置极易报错。提供了一键安装脚本 `setup_lsdt.sh`。

### 🛠️ 步骤 1：运行一键配置脚本

在根目录打开终端，运行：

```bash
bash setup_lsdt.sh
```

*注：该脚本会自动创建名为 `mujoco_py37` 的 Conda 环境，下载 MuJoCo 2.1.0 引擎，配置环境变量，并安装所有依赖包（首次安装 `mujoco-py` 会触发 GCC 编译，请等待 2-3 分钟）。*

### 🛠️ 步骤 2：激活环境

脚本运行结束后，为了让物理引擎的环境变量彻底生效，请手动执行以下两行：

```bash
source ~/.bashrc
conda activate mujoco_py37
```

> **常见报错排查 (Troubleshooting):**
>
>   * 报错: `gcc: error trying to exec 'cc1plus'`
>       * 解决: 系统缺少 C++ 编译器。运行 `sudo apt-get install g++`。
>   * 报错: `Exception: Missing path to your environment variable. MUJOCO_GL...`
>       * 解决: 说明你在无图形界面的服务器上运行，请在终端输入 `export MUJOCO_GL=egl`。
>   * 报错: `ERROR: Could not build wheels for mujoco-py`
>       * 解决: 检查 Conda 环境是否真的是 Python 3.7，部分更高版本的 Python 无法兼容旧版 `mujoco-py`。

-----

## 3\. 数据准备 (Data Preparation)

离线强化学习模型需要先“吃”进专家数据。我们需要将 D4RL 官方提供的单步交互数据，转化为 Transformer 所需的“轨迹序列 (Trajectories)”。

### 🛠️ 步骤：运行数据转换脚本

```bash
python convert_data.py
```

  * **执行效果**: 脚本会自动通过 D4RL 下载对应的环境数据（需要网络畅通），对其进行分段、计算 Return-to-go (RTG)，并最终在项目根目录的 `data/` 文件夹下生成处理好的 `.pkl` 文件。

>  **常见报错排查 (Troubleshooting):**
>
>   * 报错**: `urllib.error.URLError: <urlopen error [Errno 111] Connection refused>`
>       * 解决: D4RL 数据集托管在海外服务器。如果你在国内服务器运行，请确保配置了科学上网代理，或者手动将 `.hdf5` 数据集下载后放到 `~/.d4rl/datasets/` 目录下。

-----

## 4\. 模型训练 (Model Training)

核心的架构逻辑（Pre-LN + Sigmoid 门控）全部封装在 `decision_transformer/LSDT.py` 中，你只需要通过主脚本调兵遣将。

### 🛠️ 步骤 1：启动单环境训练 (以 Walker2d 为例)

```bash
PYTHONPATH=. python scripts/train.py --env walker2d --dataset medium --dataset_dir data
```

### 🛠️ 步骤 2：多显卡并行训练 / 跑消融实验 (进阶)

如果你有多张显卡，可以通过 `CUDA_VISIBLE_DEVICES` 指定 GPU 并在后台并行运行。如果是windows下面的虚拟机就只能用cpu逐个运行
例如，测试不同长短记忆特征通道比（分配给 GPU 1）：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/train.py --env walker2d --dataset medium --dataset_dir data --convdim 32 --log_dir AntMaze/ablation_c32/
```
这里只显示测试walker2d，最好还是训练一下hopper和halfcheetah，分别设置不同的通道比（32、64、96）和卷积核（3、5、11）进行训练，我的消融结果是在walker2d下的，如图ablation_convdim.jpg和ablation_kernel.jpg
  * **执行效果**: 训练会在 `AntMaze/` 文件夹下自动生成两个重要文件：
    1.  `[env]_lsdt.csv`：实时更新的成绩单（每 500 步记录一次）。
    2.  `[env]_lsdt_best.pt`：动态保存的历史最高分模型权重。

-----

## 5\. 模型评估 (Evaluation)

用独立脚本对训练好的模型计算多次测试的均值和方差

### 🛠️ 步骤：运行考核脚本

默认跑 10 个回合（Episodes）：

```bash
PYTHONPATH=. python evaluate_model.py --env walker2d --num_eval_episodes 10
```
也可以设置多个回合
-----

## 6\. 可视化与绘图 (Visualization)

论文和报告所需的所有高清插图，均可通过自动化脚本一键生成。**请确保 `AntMaze/` 目录下有相应的 `.csv` 日志文件或 `.pt` 模型文件。**

直接在终端运行以下命令，图片会自动保存在项目根目录下：

1.  **生成环境性能对比柱状图**：
    ```bash
    python plot_bar.py
    ```
2.  **生成真实训练收敛曲线 (Learning Curve)**：
    ```bash
    python plot_learning_curve.py
    ```
3.  **生成消融实验对比折线图 (感受野与长短特征比)**：
    ```bash
    python plot_ablation_1.py
    ```
4.  **提取并绘制内部注意力矩阵热力图 (Attention Heatmap)**：
    ```bash
    python plot_heatmap.py
    ```