https://xbpeng.github.io/projects/ASE/index.html

### 1、安装
这个代码是基于isaac gym, rl-games
- 首先安装isaac gym
- pip install -r requirements.txt
注意，这个代码支持rl-games==1.1.4，更高版本的rl-games程序不支持

### 2、运行指令

- 动作数据的可视化
```
python ase/run.py --test  \
--task HumanoidViewMotion  \
--num_envs 2  \
--cfg_env ase/data/cfg/humanoid_sword_shield.yaml  \
--cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml  \
--motion_file  ase/data/motions/reallusion_sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy
```

- AMP训练
```bash
python ase/run.py --task HumanoidAMP \
        --cfg_env ase/data/cfg/humanoid.yaml \
        --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml \
        --motion_file ase/data/motions/amp_humanoid_run.npy \
        --headless
```


- AMP 查看
```bash
python ase/run.py --test --task HumanoidAMP \
       --num_envs 16 --cfg_env ase/data/cfg/humanoid.yaml \
       --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml \
       --motion_file ase/data/motions/amp_humanoid_walk.npy \
       --checkpoint /home/fdd/fei/AMP_Human/ASE/output/Humanoid_06-15-42-42/nn/Humanoid.pth
```




### 3、debug

在run.py的200行，main()下进行如下更改，就可以debug程序了
``` python
set_np_formatting()
args = get_args()
args.task = 'HumanoidAMP'
args.cfg_env = '/home/fdd/fei/AMP_Human/ASE/ase/data/cfg/humanoid.yaml'
args.cfg_train = '/home/fdd/fei/AMP_Human/ASE/ase/data/cfg/train/rlg/amp_humanoid.yaml'
args.motion_file = '/home/fdd/fei/AMP_Human/ASE/ase/data/motions/amp_humanoid_walk.npy'
```
 
 
 <style>
 h1,h2 {text-align:center}
 p {text-indent:2em;}
	
body, th, td, .inline-code {
    font-size: 24px;
   }
h2 {
    background-color: #46515c;
    color:white;
}
</style> 