# Multi-Stage Hybrid Training for Humanoid Robot Ladder Climbing

基于 Webots 仿真平台的人形机器人斜梯攀爬分层强化学习训练框架。

## 概述

本项目在 Webots 2021a 仿真环境中，以 Robotis-OP2 人形机器人为研究对象，将斜梯攀爬这一复杂任务拆解为 **抓取（Grasping）** 与 **踩踏（Stepping）** 两个关键子任务，分别训练独立的强化学习智能体，并通过串行衔接完成完整攀爬流程。

## 仿真平台与机器人

| 项目       | 说明                                               |
| ---------- | -------------------------------------------------- |
| 仿真平台   | Webots 2021a                                       |
| 机器人模型 | Robotis-OP2                                        |
| 机械改造   | 在机器人手部增加了双指夹持器，用于适配梯级抓取任务 |
| 斜梯场景   | 10 级、45° 斜梯，梯级间距与高度均为 0.07 m        |

## 方法思路

完整斜梯攀爬任务涉及抓取、抬腿、踩踏、姿态调节等多个连续动作，直接进行端到端训练面临状态维度高、奖励稀疏、训练不稳定等问题。本项目的核心思路是 **任务分解与分阶段训练**：

### 第一阶段：抓取（Catching）

- **目标**：控制机器人双臂和夹爪逐步靠近目标梯级，并在合适姿态下形成稳定抓取。
- **状态输入**：128×128 灰度图像 + 4 维简化关节状态 + 19 节点图结构关节状态
- **网络结构**：CNN + MLP + GNN 三路特征提取后融合
- **动作输出**：2 个离散抓取动作
- **执行模块**：`python_scripts/<ALGO>/RobotRun1.py`

### 第二阶段：踩踏（Stepping）

- **目标**：在抓取成功的前提下，控制左腿完成抬腿、对准和踩踏，使脚部接触目标梯级。
- **状态输入**：20 维机器人关节状态
- **网络结构**：MLP（多层全连接网络）
- **动作输出**：4 个离散踩踏动作
- **执行模块**：`python_scripts/<ALGO>/RobotRun2.py`
- **成功判定**：脚部触碰目标梯级、距离小于 0.1 m 且持续约 2 s

### 阶段衔接

第二阶段不是独立启动，而是在第一阶段检测到抓取成功后自动触发，复用同一个 `Environment` 实例。

## 支持的强化学习算法

本项目实现了三种主流算法，均支持在两阶段训练中切换使用：

| 算法 | 特点                                           |
| ---- | ---------------------------------------------- |
| DQN  | 基于 Q-Learning，使用经验回放和目标网络        |
| PPO  | 基于策略梯度，使用 Clipped Surrogate Objective |
| SAC  | 基于最大熵框架的 Actor-Critic 方法             |

算法切换通过 `controllers/Train_main/Train_main.py` 中的注释完成。

## 项目结构

```
Multi-Stage_Hybrid_Training/
├── controllers/
│   └── Train_main/
│       └── Train_main.py          # 项目训练入口
├── python_scripts/
│   ├── Project_config.py          # 全局配置（路径、超参数、关节限制等）
│   ├── Webots_interfaces.py       # Webots 与训练代码的桥接层
│   ├── DQN_Log_write.py           # 日志写入器
│   ├── DQN/
│   │   ├── DQN_episoid.py         # 第一阶段训练调度
│   │   ├── DQN_episoid_2.py       # 第二阶段训练调度
│   │   ├── DQN_DQNnet.py          # 第一阶段 Q 网络（CNN+MLP+GNN）
│   │   ├── DQN_DQNnet_2.py        # 第二阶段 Q 网络（MLP）
│   │   ├── Replay_memory.py       # 第一阶段经验回放池
│   │   ├── Replay_memory_2.py     # 第二阶段经验回放池
│   │   ├── RobotRun1.py           # 第一阶段动作执行器
│   │   ├── RobotRun2.py           # 第二阶段动作执行器
│   │   └── Data_fusion.py         # 多模态数据融合
│   ├── PPO/                       # PPO 实现（结构与 DQN 目录同构）
│   └── SAC/                       # SAC 实现（结构与 DQN 目录同构）
├── worlds/                        # Webots 场景文件
├── libraries/                     # Robotis-OP2 运动控制库
└── README.md
```

## 快速开始

### 环境要求

- Windows / Linux / macOS
- Webots 2021a 或更高版本
- Python 3.7+
- PyTorch（GPU 推荐，CPU 可用）
- 其他依赖见 `python_scripts/` 下各模块 import

### 运行步骤

1. 用 Webots 打开 `worlds/` 目录下的场景文件。
2. 修改 `python_scripts/Project_config.py` 中的路径配置，使其匹配你的本地目录结构。
3. 在 `controllers/Train_main/Train_main.py` 中通过注释切换算法（DQN / PPO / SAC）。
4. 在 Webots 中启动仿真，训练自动开始。

### 关键配置项

`python_scripts/Project_config.py` 中可调整的主要参数：

- `device`：`cuda` 或 `cpu`
- `BATCH_SIZE`、`LR`、`EPSILON`、`GAMMA`、`MEMORY_CAPACITY`：DQN 超参数
- `LR_ACTOR`、`LR_CRITIC`、`TAU`：SAC 超参数
- `gps_goal` / `gps_goal1`：抓取与踩踏目标位置
- `Darwin_config`：机器人关节限制、稳定性阈值等

## 实验结果

| 算法 | 抓取 Episodes | Return 均值 | 抓取成功率 | 踩踏 Episodes | 踩踏成功率 |
| ---- | ------------: | ----------: | ---------: | ------------: | ---------: |
| DQN  |        24,781 |        7.55 |     22.95% |         5,687 |      1.69% |
| PPO  |        30,849 |        2.34 |      5.09% |         1,569 |      2.04% |
| SAC  |        31,399 |        8.95 |     19.73% |         6,194 |     27.88% |

综合来看，SAC 在踩踏阶段显著优于其他两种算法，是当前任务中整体表现最优的方案。

## 说明

- 本项目是一个研究性训练框架，训练效果仍有优化空间。
- 代码中存在实验性、不完全一致的部分（如网络输出维度与实际动作数），整体更适合作为实验平台而非生产级系统。
- 抓取阶段网络采用多模态融合设计，结合了视觉、局部状态和关节拓扑信息；踩踏阶段则以动作控制为主，网络结构相对简洁。
