## 一、代码推荐
https://github.com/isaac-sim/IsaacGymEnvs

https://github.com/SZU-AdvTech-2023/055-AMP-Adversarial-Motion-Priors-for-Stylized-Physics-Based-Character-Control


## 二、安装及运行
https://github.com/isaac-sim/IsaacGymEnvs
### 2.1 安装
- 首先安装isaac gym
- pip install -e .
注意，通过pip list可以查看 isaacgymenvs 路径

### 2.2 训练指令
在common_agent.py的78行下增加  self.seq_len = config['seq_len']， 否则会报错

python isaacgymenvs/train.py task=HumanoidAMP headless=True
python isaacgymenvs/train.py task=G1AMP headless=True

接着训练
python isaacgymenvs/train.py task=HumanoidAMP headless=True    checkpoint=/home/fdd/runs/HumanoidAMP_21-19-03-07/nn/HumanoidAMP_21-19-03-08_1000.pth
python isaacgymenvs/train.py task=G1AMP headless=True    checkpoint=/home/fdd/runs/G1AMP_02-22-48-27/nn/G1AMP_02-22-48-29_1000.pth

### 2.3查看
 
python isaacgymenvs/train.py  task=HumanoidAMP headless=False test=True num_envs=2 checkpoint=/home/fdd/runs/HumanoidAMP_21-22-06-01/nn/HumanoidAMP_21-22-06-02_400.pth
 
python isaacgymenvs/train.py  task=G1AMP headless=False test=True num_envs=2 checkpoint=/home/fdd/runs/G1AMP_27-20-41-17/nn/G1AMP_27-20-41-19_7000.pth

### 2.4 调试方法
train.py 函数下增加
``` python
import sys
sys.argv = ["launch_rlg_hydra.py", "task=HumanoidAMP", "headless=True"]
```

## 三、网络

### 3.1 连续动作空间的策略
在连续动作空间中，策略网络（Policy Network）输出一个概率分布的参数。最常见的做法是输出一个多元高斯分布（Multivariate Gaussian Distribution） 的参数：
均值 (Mean, μ): 决定了动作的“中心”或“期望”值。
标准差 (Standard Deviation, σ): 决定了动作的“随机性”或“探索程度”。σ 越大，从该分布中采样的动作越分散，探索性越强；σ 越小，动作越集中在均值附近。

这里的sigma是固定的，action 网络的std 两种处理方法：
- AMP
```python
self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)  
std = torch.exp(log_std)  
dist = Normal(mu, std)
```   

- rsl_rl
```python
self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))   init_noise_std = 1
```

通过学习log_std 更好
![bb35a6b9e4603ee683b563a783aaf76a.png](../../_resources/bb35a6b9e4603ee683b563a783aaf76a.png)

### 3.2 horizon_length 是什么？

horizon_length 是指在一次 rollout（轨迹收集）中，智能体在更新策略之前所执行的动作序列长度。
 
换句话说：

它定义了 每个 episode（或 trajectory）的最大步数（steps）；
或者说，它是 策略更新前，智能体与环境交互的步数。

在 PPO 中，horizon_length 和其他参数一起决定了：

每个策略更新前，智能体收集多少数据；
数据总量 = num_actors * horizon_length 

每次更新使用这些数据进行多次 minibatch 训练（由 mini_epochs 控制）。

episodeLength 强制重置仿真器的时间

### 3.3 disc 网络
"disc_agent_logit"	当前 Agent 的动作	鉴别器对当前策略生成动作的打分	用于计算损失，鼓励 Agent 生成更自然的动作
"disc_agent_replay_logit"	Replay Buffer 中的历史动作	鉴别器对旧策略动作的打分	提高训练稳定性，防止过拟合
"disc_demo_logit"	专家示范数据（motion capture）	鉴别器对专家动作的打分	作为正样本，帮助鉴别器学习“自然动作”的分布
 
### 3.4 bound loss

你提出的这段代码实现的是一种 Bound Loss（边界损失），也常被称为 Clipped Action Penalty 或 Action Regularization，在强化学习（特别是 PPO 等策略梯度方法）中用于鼓励策略网络输出的动作不要过于接近动作空间的边界。
```
soft_bound = 1.0
mu_loss_high = torch.maximum(mu - soft_bound, torch.tensor(0, device=self.ppo_device)) ** 2
mu_loss_low  = torch.minimum(mu + soft_bound, torch.tensor(0, device=self.ppo_device)) ** 2
b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
```
如果 mu 非常接近 +1 或 -1，即使 sigma > 0，采样出的动作也几乎总是被 clipped 到边界值，导致：

探索能力下降：动作被“卡”在边界，无法有效探索边界附近的真实最优值。
梯度误导：网络误以为“输出 +1 是最优的”，但实际上可能 0.95 才是最佳。
训练不稳定：边界附近的动作可能导致物理仿真不稳定（如关节锁死、巨大力矩）
bounds_loss_coef = 10 

### 3.5 网络保存
 amp_players.py 的 def restore(self, fn):
```
path = os.path.join('runs', 'policy_1.pt')
        model = torch.nn.Sequential(
            copy.deepcopy(self.model.a2c_network.actor_mlp),
            copy.deepcopy(self.model.a2c_network.mu)
                ).eval().to('cpu') 
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path) 

        norm_state = self.model.running_mean_std.state_dict()
        mean = norm_state['running_mean'].cpu().numpy()
        std = torch.sqrt(norm_state['running_var'] + 1e-5).cpu().numpy() 
        import numpy as np
        np.savez('runs/obs_norm.npz', mean=mean, std=std)
```

这里还要保存 normalize_input 的相关数据，onnx的处理

```
data = np.load('/home/fdd/fei/AMP_Human/AMP2/AMP_TEST/sim2sim_onnx/config/obs_norm.npz')
    mean_obs = data['mean'].astype(np.float32)   # (71,)
    std_obs = data['std'].astype(np.float32)     # (71,)
normalized_obs = (obs_buff - mean_obs) / std_obs
cl_obs = np.clip(normalized_obs, -5, 5)
```

## 四、运行过程

### 4.1 过程

train.py-- runner.run

torch_runner.py-- run_train 首先初始化agent，再agent.train，agent 定义为amp_continuous.py

common_agent.py -- train()

amp_continuous.py -- train_epoch() 
-- play_steps() 和环境交互horizon_length步 

vec_task.py --  reset_done 重置环境，会采集数据
					--  step 和仿真交互

amp_continuous.py -- 数据 mini_epochs_num更行 calc_gradients(self, input_dict)

### 4.2 创建环境

 pd 模式
这个代码通过更改成xml就可以运行了

mjcf 的 xml 文件，要将limit 设置成true，这个会在程序里设置scale，这里我手动改了


### 4.3 obs  
机器人自由度数为28，obs中会将3自由度转换为6自由度的norm和tan矢量，就是52自由度

```
    # RM here is the observation
    # z height: 1
    # root rot (6d): 6
    # local root lin vel: 3
    # local root ang vel: 3
    # dof pos (convert to quat rotation): 13*4 = 52
    # dof vel: 28
    # end effector pose: 12  (see self._key_body_ids == [5, 8, 11, 14])
```

reward


num_amp_obs = self._num_amp_obs_steps * NUM_AMP_OBS_PER_STEP
这里有2步， 
NUM_AMP_OBS_PER_STEP = 13 + 52 + 28 + 12 # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]

### 4.4 起始状态 
参考数据选取的初始化，这里选择的随机采样参考轨迹数据

推荐文章 https://blog.csdn.net/heng6868/article/details/149069821
