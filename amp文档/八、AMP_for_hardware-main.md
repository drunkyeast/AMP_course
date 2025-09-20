

https://github.com/escontra/AMP_for_hardware


## 安装训练
conda activate humanoidGym

以前安装过，所以先卸载
pip uninstall rsl_rl 
pip uninstall legged_gym


安装
cd rsl_rl
pip install -e .

cd .. 
pip install -e .
 

python legged_gym/scripts/train.py --task=a1_amp --headless --num_envs=2048 
python legged_gym/scripts/play.py --task=a1_amp  --checkpoint  36450
 
 
## AMP_for_hardware-main
通过这里的程序可以查看 a1data2.mp4数据

转向各一个数据，只包含转90度的数据，
行走为直线行走，大概10多步

这个程序和之前的最大的不同，就是在rsl_rl 不同

这里的狗obs没有历史数据，只有48维度

``` python
# conda activate rospy38
import os
import sys
import time
 
# 从文件读取数据
import json
import numpy as np

dataPath = "/home/fdd/fei/AMP_Human/AMP_for_hardware-main/datasets/mocap_motions"
fileNameList = os.listdir(dataPath)
for ii in fileNameList:
    data_target = dataPath+'/' + ii # fileNameList[0]
 
    with open(data_target, "r") as f:
        motion_json = json.load(f) 
        motion_s1 = np.array(motion_json['Frames'])  # (len, 61)
    motion_s = motion_s1 #[::10, :]

    root_pos = motion_s[:, 0:3]  # 3
    root_quat = motion_s[:, 3:7]  # 4   
    joint_pos = motion_s[:, 7:19]  # 12
    foot_pos  = motion_s[:, 19:31]  # 12

    lin_vel = motion_s[:, 31:34]  # 3  
    ang_vel = motion_s[:, 34:37]  # 3  
    joint_vel = motion_s[:, 37:49]  # 12
    foot_vel = motion_s[:, 49:61]  # 12 
    
 
 

```








