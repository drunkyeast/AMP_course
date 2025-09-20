
### 生成pt
### 转换onnx
### 运行gym
sim2sim_gym_amp/issacGymHumanoidPD_g1walk_command.py


## 关于仿真中用的常用数据，坐标系说明

### mujoco 获取的速度

q = d.qpos.copy() # 获取 base的xyz 位置世界坐标系，四元数wxyz相对世界系，关节角度
dq = d.qvel.copy() # base的xyz速度世界坐标系，base角速度xyz在body坐标系，关节速度

``` python
root_position = q[:3]  # world
root_quat_wxyz = q[3:7]  # wxyz
root_quat_xyzw = root_quat_wxyz[[1,2,3,0]]  # xyzw
joint_angle_pos = q[7:] 

# DQ
root_world_lineVel = dq[:3] # world
rota = R.from_quat(root_quat_xyzw)
root_body_lineVel = rota.apply(root_word_lineVel, inverse=True)  # In the body frame
root_body_angVel = dq[3:6]   # In the body frame
joint_angle_vel = dq[6:] 
```

### pinocchio的定义

q = d.qpos.copy() # 获取 base的xyz 位置世界坐标系，四元数xyzw相对世界系，关节角度
dq = d.qvel.copy() # base的xyz速度body坐标系，base角速度xyz在body坐标系，关节速度

``` python
root_position = q[:3]  # world
root_quat_xyzw = q[3:7]  # xyzw 
joint_angle_pos = q[7:] 

# DQ
root_body_lineVel = dq[:3] # In the body frame
rota = R.from_quat(root_quat_xyzw)
root_world_lineVel = rota.apply(root_body_lineVel, inverse=False) # world
root_body_angVel = dq[3:6]   # In the body frame
joint_angle_vel = dq[6:] 
```

### issac gym

root_states # 获取 base的xyz 位置世界坐标系，四元数xyzw相对世界系，root线速度世界坐标系，角速度世界坐标系

base的线速度和角速度都是世界坐标系的，需要转换到body

``` python
base_pos = self.root_states[:, 0:3]
base_quat = self.root_states[:, 3:7]

root_world_lin_vel = self.root_states[:, 7:10]
root_world_ang_vel = self.root_states[:, 10:13]
base_body_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
base_body_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

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
