import mujoco, mujoco_viewer
import numpy as np
import onnxruntime
import math
from collections import deque
import os
from scipy.spatial.transform import Rotation as R
from types import SimpleNamespace
import yaml
import pygame
from threading import Thread

KEY_BODY_NAMES = ["head_link_o", "left_hand_link",
                  "right_hand_link", "left_ankle_roll_link", 
                    "right_ankle_roll_link"]

x_vel_cmd,y_vel_cmd,yaw_vel_cmd = 1.0, 0.0, 0.0 
x_vel_max,y_vel_max,yaw_vel_max = 3.5, 1.0, 1.0
stand_walk_flag = 0
 
joystick_use = True
joystick_opened = False
class cmd:
    vx = 1.0
    vy = 0.0
    dyaw = 0.0
if joystick_use:
    pygame.init()
    try:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"cannot open joystick device:{e}")

    exit_flag = False

    def handle_joystick_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd, stand_walk_flag
        
        
        while not exit_flag:
            pygame.event.get()

            x_vel_cmd = -joystick.get_axis(1) * x_vel_max
            y_vel_cmd = -joystick.get_axis(0) * y_vel_max
            yaw_vel_cmd = -joystick.get_axis(3) * yaw_vel_max
       
            aa = np.array([x_vel_cmd,y_vel_cmd,yaw_vel_cmd])
            if  np.linalg.norm(aa) < 0.1  :
                x_vel_cmd = 0
                y_vel_cmd = 0
                yaw_vel_cmd = 0
            pygame.time.delay(100)
            print(f"x_vel_cmd: {x_vel_cmd}----y_vel_cmd: {y_vel_cmd}---y_vel_cmd: {yaw_vel_cmd}---stand_walk_flag: {stand_walk_flag}")


    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()
 

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])
 

def get_mujoco_data(data,model,debug_mj_data):
    mujoco_data={}
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = np.array([q[4], q[5], q[6], q[3]]) # 直接获取 xyzw 
    # quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double) # 传感器
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # base
    root_angvel = dq[3:6]   # base
    # root_angvel = data.sensor('angular-velocity').data.astype(np.double) # 传感器
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    root_euler = quaternion_to_euler_array(quat)
    root_euler[root_euler > math.pi] -= 2 * math.pi

    mujoco_data['mujoco_dof_pos'] = q[7:]
    mujoco_data['mujoco_dof_vel'] = dq[6:] 
    mujoco_data['mujoco_root_euler'] = root_euler

    mujoco_data['mujoco_root_pos'] = q[:3]
    mujoco_data['mujoco_root_rot'] = quat  # xyzw
    mujoco_data['mujoco_root_vel'] = data.qvel[:3] # world
    mujoco_data['mujoco_root_angVel'] = r.apply(dq[3:6], inverse=False).astype(np.double)  # world

    key_pos = []
    for body_name in  KEY_BODY_NAMES: # 替换为你想查询的 link 名称
        body_id = model.body(name=body_name).id
        body_pos = data.xpos[body_id]  # shape: (3,) 世界坐标系下的 [x, y, z]
        key_pos.append(body_pos)
        print(f"Body '{body_name}' position: {body_pos}")
    key_pos = np.array(key_pos)
    mujoco_data['mujoco_key_pos'] = key_pos #.reshape(-1)  

    return mujoco_data
 

def calc_heading(q):
    # calculate heading direction from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    ref_dir = np.zeros_like(q[0:3])
    ref_dir[0] = 1
    rot_dir = my_quat_rotate(q, ref_dir)

    heading = np.arctan2(rot_dir[1], rot_dir[0])
    return heading

def normalize(x, eps=1e-8):
    # 计算 L2 范数，沿最后一个维度
    norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    # 防止除以 0，clamp 最小值
    norm = np.maximum(norm, eps)
    # 归一化
    return x / norm

def quat_unit(a):
    return normalize(a)
import numpy as np

def normalize(x, eps=1e-8):
    norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm

def quat_from_angle_axis(angle, axis):
    """
    从旋转角和旋转轴生成四元数 (w, x, y, z)
    angle: float 或 np.float64（标量）
    axis: 一维数组，如 (3,)
    """
    # 确保 angle 是 Python 标量或可用 np.sin 的类型
    sin_half = np.sin(angle / 2.0)
    cos_half = np.cos(angle / 2.0)

    # 归一化旋转轴
    axis_normalized = normalize(axis)  # shape: (3,)
    
    # 构造四元数
    xyz = axis_normalized * sin_half
    w = cos_half

    # 返回 [w, x, y, z]
    return quat_unit(np.concatenate([ xyz,np.array([w])], axis=-1))

 

def calc_heading_quat_inv(q):
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = np.zeros_like(q[0:3])
    axis[2] = 1

    heading_q = quat_from_angle_axis(-heading, axis)
    return heading_q
def my_quat_rotate(q, v):
    shape = q.shape
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.matmul(q_vec, v) * 2.0
    return a + b + c

def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[0], a[1], a[2], a[3]
    x2, y2, z2, w2 = b[0], b[1], b[2], b[3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = np.stack([x, y, z, w], dim=-1).view(shape)

    return quat
def quat_to_tan_norm(q):
    # represents a rotation using the tangent and normal vectors
    ref_tan = np.zeros_like(q[0:3])
    ref_tan[0] = 1
    tan = my_quat_rotate(q, ref_tan)
    
    ref_norm = np.zeros_like(q[0:3])
    ref_norm[-1] = 1
    norm = my_quat_rotate(q, ref_norm)
    
    norm_tan = np.concatenate([tan, norm] )
    return norm_tan

def compute_humanoid_observations(
        cfg,mujoco_data,  
         local_root_obs): 

    root_pos = mujoco_data['mujoco_root_pos'] 
    root_rot = mujoco_data['mujoco_root_rot'] # xyzw
    root_vel = mujoco_data['mujoco_root_vel']  # world
    root_ang_vel = mujoco_data['mujoco_root_angVel']# world
    key_body_pos = mujoco_data['mujoco_key_pos']
    dof_pos = mujoco_data['mujoco_dof_pos']
    dof_vel = mujoco_data['mujoco_dof_vel'] 

    root_h = root_pos[2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):  # false
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)  # quat -> 6d

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos 
    local_key_body_pos = key_body_pos - root_pos_expand
    
    local_end_pos = local_key_body_pos.copy()

    for i in range(local_key_body_pos.shape[0]):
        local_end_pos[i,:] = my_quat_rotate(heading_rot, local_key_body_pos[i, :])

    flat_local_key_pos = local_end_pos.reshape(-1) 

    dof_obs = dof_pos

    # RM here is the observation
    # z height: 1
    # root rot (6d): 6
    # local root lin vel: 3
    # local root ang vel: 3
    # dof pos (convert to quat rotation): 13*4 = 52
    # dof vel: 28
    # end effector pose: 12  (see self._key_body_ids == [5, 8, 11, 14])
    command = np.array([x_vel_cmd,y_vel_cmd,yaw_vel_cmd])
    command *= (np.linalg.norm(command ) > 0.1) 
    obs = np.concatenate((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos,
                          command) )
    return   obs.reshape(1, -1).astype(np.float32)  # (1, D)


def pd_control(target_pos,dof_pos, target_vel,dof_vel ,cfg):
    '''Calculates torques from position commands
    '''
    torque_out = (target_pos  - dof_pos ) * cfg.kps + (target_vel - dof_vel)* cfg.kds
    return torque_out

def run_mujoco(cfg,meanStdPath):
    # 加载参数
    data = np.load(meanStdPath)
    mean_obs = data['mean'].astype(np.float32)
    std_obs = data['std'].astype(np.float32) 

    # mujoco接口初始化
    model = mujoco.MjModel.from_xml_path(cfg.xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = cfg.simulation_dt
    data.qpos[-cfg.num_actions:] = cfg.default_dof_pos * 0
    
    # 进行物理仿真
    mujoco.mj_step(model, data)

    # mujoco可视化设置
    viewer = mujoco_viewer.MujocoViewer(model,data) 
    viewer.cam.distance=3.0 
    viewer.cam.azimuth = 90
    viewer.cam.elevation=-45
    viewer.cam.lookat[:]=np.array([0.0,-0.25,0.824])

    # 策略模型加载
    onnx_model_path = cfg.policy_path
    policy = onnxruntime.InferenceSession(onnx_model_path)

    # 变量初始化
    target_dof_pos = np.zeros((1,cfg.num_actions), dtype=np.double)
    target_dof_vel = np.zeros((1,cfg.num_actions), dtype=np.double)
    action = np.zeros((1,cfg.num_actions), dtype=np.double)

    count = 0

    debug_mj_data = {}
    mujoco_data_all_dict = {}
    tar_pos = [0, 0, 0]
    target_dof_pos_list = np.zeros((1, cfg.num_actions)).tolist()
    mujoco_data_all_dict["tar_pos"] = []
    mujoco_data_all_dict["target_dof_pos"] = []

    # 策略模型onnx推理
    _pd_action_offset = np.array([-0.1,0.0,0.0,0.3,-0.2,0.0,-0.1,0.0,0.0,0.3,-0.2,0.0,0.0,0.0,0.0,0.0,0.2,0.0,1.1,0.0,-0.2,0.0,1.1])
    _pd_action_scale = np.array([3.7873, 2.4435, 3.8606, 2.0769, 0.9774, 0.3665, 3.7873, 2.4435, 3.8606,
                                 2.0769, 0.9774, 0.3665, 3.6652, 0.7280, 0.7280, 4.0317, 2.6878, 3.6652,
                                 2.1991, 4.0317, 2.6878, 3.6652, 2.1991] )
    
    ## 执行回合
    for time_cnt in range(int(cfg.sim_duration / cfg.simulation_dt)):
        mujoco_data = get_mujoco_data(data,model,debug_mj_data)
        
        ## 控制频率
        if count % cfg.decimation == 0: 
            obs_buff = compute_humanoid_observations(cfg, mujoco_data, False) 
            normalized_obs = (obs_buff - mean_obs) / std_obs
            cl_obs = np.clip(normalized_obs, -5, 5)
            policy_input = {policy.get_inputs()[0].name: cl_obs}
            action = policy.run(["output"], policy_input)[0]
            action = np.clip(action, -cfg.clip_actions, cfg.clip_actions)
            target_dof_pos = action * _pd_action_scale + _pd_action_offset
 
        # Generate PD control
        tau = pd_control(target_dof_pos, mujoco_data["mujoco_dof_pos"], 
                        target_dof_vel, mujoco_data["mujoco_dof_vel"], cfg)
        
        target_dof_pos_list = target_dof_pos[0].tolist()
        mujoco_data_all_dict["target_dof_pos"].append(target_dof_pos_list.copy())  

        tau = np.clip(tau, -cfg.tau_limit, cfg.tau_limit) 
        data.ctrl = tau
        mujoco.mj_step(model, data)
        viewer.render()
        count += 1
 
        tar_pos[0]  += x_vel_cmd * cfg.simulation_dt
        tar_pos[1]  += y_vel_cmd * cfg.simulation_dt
        tar_pos[2]  += yaw_vel_cmd * cfg.simulation_dt

        for k,v in debug_mj_data.items():
            if k not in mujoco_data_all_dict:
                mujoco_data_all_dict[k] = [v]
            else:
                mujoco_data_all_dict[k].append(v)

        mujoco_data_all_dict["tar_pos"].append(tar_pos.copy())
        # if time_cnt == 8000:
    import json
    with open("mujoco_data_all_dict.json", "w") as mujoco_file:
        json.dump(mujoco_data_all_dict, mujoco_file, indent=4)

    print("-----save-----")
    viewer.close()

def update_yaml_value(file_path, key, new_value, key1, new_value1):
    with open(file_path, "r") as f:
        data = yaml.safe_load(f) or {}
    data[key] = new_value  
    data[key1] = new_value1
    with open(file_path, "w") as f:
        yaml.dump(data, f)
    print(f"Updated {key} = {new_value}")


if __name__ == "__main__":
    import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("model_name",  type=str, default='0')
    # args = parser.parse_args() 
    policy_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/amp_4080_walk_14400.onnx') 
    xml_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robot/g1_description/mjcf/g1_29dof_anneal_23dof_mujoco.xml')
 
    meanStdPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/obs_norm_14400.npz')
    cfg = SimpleNamespace() 
 
    cfg.simulation_dt = 0.00166   
    cfg.default_dof_pos = np.array([-0.1,0.0,0.0,0.3,-0.2,0.0,-0.1,0.0,0.0,0.3,-0.2,0.0,0.0,0.0,0.0,0.0,0.2,0.0,1.1,0.0,-0.2,0.0,1.1])
    cfg.kds = np.array([2.5, 2.5, 2.5, 5.0, 0.2, 0.2, 2.5, 2.5, 2.5, 5.0, 0.2, 0.2, 5.0, 5.0, 5.0, 2.0, 2.0, 0.1, 1.0, 2.0, 2.0, 0.1, 1.0])
    cfg.kps = np.array([100.0, 100.0, 100.0, 200.0, 40.0, 40.0, 100.0, 100.0, 100.0, 200.0, 40.0, 40.0, 400.0, 400.0, 400.0, 90.0, 60.0, 20.0, 60.0, 90.0, 60.0, 20.0, 60.0])
    cfg.xml_path = xml_path 
    cfg.num_actions = len(cfg.kps)
    cfg.policy_path = policy_path
    cfg.sim_duration = 100   
    cfg.decimation = 20 
    cfg.clip_actions = 100
    cfg.tau_limit =np.array([ 74.8, 74.8, 74.8, 118.15, 42.5, 42.5, 74.8, 74.8, 74.8, 118.15, 42.5, 42.5, 74.8, 42.5, 42.5, 21.25, 21.25, 21.25, 21.25, 21.25, 21.25, 21.25, 21.25])
    
    run_mujoco(cfg, meanStdPath)

    print("----done-----")   

