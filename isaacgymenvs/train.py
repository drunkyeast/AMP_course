# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import hydra

from omegaconf import DictConfig, OmegaConf
from omegaconf import DictConfig, OmegaConf


def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict['params']['config']

    train_cfg['device'] = cfg.rl_device

    train_cfg['population_based_training'] = cfg.pbt.enabled
    train_cfg['pbt_idx'] = cfg.pbt.policy_idx if cfg.pbt.enabled else None

    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    print(f'Using rl_device: {cfg.rl_device}')
    print(f'Using sim_device: {cfg.sim_device}')
    print(train_cfg)

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg") # 这个装饰器会拦截函数体, 具体干什么略, 没必要看, 垃圾代码, 是给人看的吗?
def launch_rlg_hydra(cfg: DictConfig):

    import logging
    import os
    from datetime import datetime

    # noinspection PyUnresolvedReferences
    import isaacgym
    from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
    from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank
    from hydra.utils import to_absolute_path
    from isaacgymenvs.tasks import isaacgym_task_map
    import gym
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed

    if cfg.pbt.enabled:
        initial_pbt_check(cfg)

    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs


    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    def create_isaacgym_env(**kwargs): 
        envs = isaacgymenvs.make( # 创建环境, 这时才会出现isaacgym窗口
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    env_configurations.register('rlgpu', { # 我就说很奇怪, 为什么单步调试不进去. 断点被过滤器排除: env_configurations.register() 来自外部库 rl_games，调试器认为这不是"你的代码"，所以跳过了断点。有一个justMycode的参数设置, 具体不清楚, 略.
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs), # 这个是lambda表达式, 不会执行. 这是注册环境, 不知何时才会创建
    })

    ige_env_cls = isaacgym_task_map[cfg.task_name]
    dict_cls = ige_env_cls.dict_obs_cls if hasattr(ige_env_cls, 'dict_obs_cls') and ige_env_cls.dict_obs_cls else False

    if dict_cls:
        
        obs_spec = {}
        actor_net_cfg = cfg.train.params.network
        obs_spec['obs'] = {'names': list(actor_net_cfg.inputs.keys()), 'concat': not actor_net_cfg.name == "complex_net", 'space_name': 'observation_space'}
        if "central_value_config" in cfg.train.params.config:
            critic_net_cfg = cfg.train.params.config.central_value_config.network
            obs_spec['states'] = {'names': list(critic_net_cfg.inputs.keys()), 'concat': not critic_net_cfg.name == "complex_net", 'space_name': 'state_space'}
        
        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec, **kwargs))
    else:

        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs)) # 双重注册, 这而才跳回去创建环境, 卧槽

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    observers = [RLGPUAlgoObserver()]

    if cfg.pbt.enabled:
        pbt_observer = PbtAlgoObserver(cfg)
        observers.append(pbt_observer)

    if cfg.wandb_activate:
        cfg.seed += global_rank
        if global_rank == 0:
            # initialize wandb only once per multi-gpu run
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer) # 这儿会创建一个Runner, 里面有两个工厂的创建!!!, 会默认注册一些A2C和SAC算法的东西.
        # 这就是工厂模式的经典执行流程！让我解释一下这个"奇怪"的逻辑：. 注册阶段 (之前执行过的):  调用阶段 (你现在看到的):
        """
        工厂模式的精髓：
        传统方式：
        # 硬编码，不灵活
        if algo_name == 'amp_continuous':
            agent = AMPAgent(**kwargs)
        elif algo_name == 'ppo':
            agent = PPOAgent(**kwargs)

        工厂模式：
        # 注册阶段
        factory.register('amp_continuous', lambda **kwargs: AMPAgent(**kwargs))
        factory.register('ppo', lambda **kwargs: PPOAgent(**kwargs))
        # 使用阶段
        agent = factory.create(algo_name, **kwargs)  # 根据配置动态创建

        为什么这样设计：
        解耦 - 创建逻辑与使用逻辑分离
        扩展性 - 轻松添加新算法
        配置驱动 - 通过配置文件控制使用哪个算法
        所以你看到的"奇怪逻辑"实际上是延迟执行的工厂模式：先注册创建函数，后根据需要调用！
        这就是为什么调试时会"跳来跳去"的原因 - 这是设计模式的正常执行流程。
        """
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs)) # Agent在这里定义, 
        # 关于这个lambda表达式的理解: python中`lambda x: x+1` 等价于 C++中[]`(int x) { return x+1; }`
        # 当工厂中return builder(**kwargs) 时, 相当于给这个lambda函数传入参数, 然后执行函数体里面的amp_continuous.AMPAgent(**kwargs)这句话.
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network)) # 这儿怎么会跳转到创建环境env
        # 目前看下来, 注册顺序与执行顺序不一样, 下面这一句还先执行呢. 工厂模式, 怎么调试啊.
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers)) # 这东西有必要细看
    runner.load(rlg_config_dict) # 
    runner.reset() # 为什么pass了?

    # dump config dict
    if not cfg.test: # 所有的cfg.xxx都会涉及到OmegaConf相关东西, 不过你不用深究.
        experiment_dir = os.path.join('runs', cfg.train.params.config.name + 
        '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))

        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f: # 这个with的语法就类似C++的RAII机制, 确保文件会被关闭.
            f.write(OmegaConf.to_yaml(cfg))

    '''
    对于这个项目：
    先跑通流程：不求甚解，先让代码运行
    关注核心：重点理解强化学习算法部分
    渐进深入：遇到问题再深入框架细节

    不要有心理负担：
    很多PhD级别的算法工程师也不会深究每个框架细节
    重点是解决问题，不是理解每行代码
    分工协作比个人全栈更高效

    你现在的做法很对：
    通过调试理解执行流程
    对比C++来理解概念
    关注实际问题解决

    记住：好的工程师知道什么时候深入，什么时候跳过细节

    涉及的设计模式：
    这个项目确实用了几个设计模式：
    工厂模式 - runner.algo_factory.register_builder()
    观察者模式 - MultiObserver(observers)
    策略模式 - 不同的算法实现
    单例模式 - 环境管理
    你需要专门学设计模式吗？
    答案：不需要！
    原因：
    你已经在用了 - 通过阅读代码自然理解
    实用优于理论 - 解决问题比背概念重要
    时间成本 - 学设计模式不如专注核心目标

    你的重点应该是：
    强化学习算法 - PPO、AMP的核心原理
    IsaacGym → MuJoCo - 你的实际目标
    PyTorch → ONNX - 模型转换技术
    '''
    runner.run({ # 经典调试问题, 外部库函数 - rl_games 是第三方库。justMyCode 默认为 true - 调试器跳过第三方代码。怎么又提到了什么多观察者管理器啥的, 不懂python运行逻辑啊.
        'train': not cfg.test, # 当你访问 cfg.test、cfg.checkpoint 等属性时，实际上调用的是：# OmegaConf内部实现 def __getattr__(self, key):
        'play': cfg.test, # 你需要搞明白吗？ 你只需要知道： cfg.test = 从配置文件读取 test 参数, 其他的内部实现完全可以忽略
        'checkpoint': cfg.checkpoint, # 好的工程师知道什么时候深入，什么时候跳过细节,  OmegaConf就是个工具，用就行了，不用深究。就像你开车不需要懂发动机原理一样！
        'sigma': cfg.sigma if cfg.sigma != '' else None # ✅ 关注: 强化学习算法、模型训练逻辑 ❌ 忽略: OmegaConf内部实现、配置解析细节
    }) # 专注于解决你的实际问题：IsaacGym到MuJoCo的模型部署，这才是重点！


if __name__ == "__main__":
    import sys
    # sys.argv = ["train.py", "task=HumanoidAMP", "headless=False"]
    sys.argv = ["train.py", "task=HumanoidAMP", "headless=False", "num_envs=512", "train.params.config.minibatch_size=8192"]
    # sys.argv = ["train.py", "task=G1AMP", "headless=True"]
 
    # sys.argv = ["launch.py", "task=G1AMP", "headless=False",
    #             "test=True", "num_envs=2",  # test=True会进入推理模式, 又叫play...
    #             "checkpoint=/home/fdd/fei/AMP_Human/AMP_course/runs/G1AMP_02-23-06-14/nn/G1AMP_02-23-06-16_3000.pth"]


    launch_rlg_hydra()
