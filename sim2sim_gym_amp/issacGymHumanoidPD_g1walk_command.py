import os
import sys
import time 
# 引入 Isaac Gym 的 Python API
from isaacgym import gymapi
from isaacgym import gymtorch
from torch_jit_utils import quat_mul, to_torch, calc_heading_quat_inv, quat_to_tan_norm, my_quat_rotate

import onnxruntime
import torch
import numpy as np
import pygame  
from threading import Thread 

KEY_BODY_NAMES = ["head_link_o", "left_hand_link","right_hand_link", 
                  "left_ankle_roll_link",  "right_ankle_roll_link"]

# 这儿控制了速度.
# x_vel_cmd,y_vel_cmd,yaw_vel_cmd = 1.0, 0.0, 0.0  # 前进
x_vel_cmd,y_vel_cmd,yaw_vel_cmd = 3.0, 0.0, 0.0  # 加速前进
x_vel_max,y_vel_max,yaw_vel_max = 3.5, 1.0, 1.0
stand_walk_flag = 0
 
joystick_use = False # 我没有手柄啊, 所以禁用掉吧
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
            # 这儿是手柄的映射!!!!!!!!!!!!!!!!!!!!!
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

# ================================
# 主类：Isaac Gym 仿真示例
# ================================
def get_axis_params(value, axis_idx, x_value=0., dtype=float, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))

def build_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs,command):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
 
    dof_obs = dof_pos #dof_to_obs(dof_pos)
    # dof_obs[:,4] *= 0
    # dof_obs[:,10] *= 0

    dof_v = dof_vel
    # dof_v[:,4] *= 0
    # dof_v[:,10] *= 0
    command *= (torch.norm(command ) > 0.1) 
    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, 
                     dof_obs, dof_v, flat_local_key_pos, command.unsqueeze(0) ), dim=-1)
    return obs

class HumanoidSim:
    def __init__(self):
        self.gym = gymapi.acquire_gym()
        self.device = 'cuda:0'
        # -------------------------------
        # 仿真参数
        # -------------------------------
        self.sim_dt = 0.0166   # 1/60 
        self.num_envs = 1         # 环境数量
        self.asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robot')
        self.asset_file = "g1_description/mjcf/g1_29dof_anneal_23dof.xml"
        self.meanStdPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/obs_norm_14400.npz')
        self.onnxPath =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/amp_4080_walk_14400.onnx')
        # self.meanStdPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/obs_norm_walk_run.npz')
        # self.onnxPath =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/amp_walk_run.onnx')
        '''
        为什么模型与归一化参数分开搞呢?
        强化学习框架（如rl_games）通常将：
          模型权重：神经网络参数
          环境统计：观测归一化参数
          训练状态：优化器状态等
        分开存储，这是历史习惯。
        '''
        self.last_frame_time = 0 
        self.control_freq_inv = 2  # 这个东西决定了, 60Hz与30Hz的问题.


        # -------------------------------
        # 初始化
        # -------------------------------
        self._create_sim()
        self._create_ground_plane()
        self._create_envs()
        self.gym.prepare_sim(self.sim) 
 
        self.viewer = self.gym.create_viewer(
            self.sim, gymapi.CameraProperties())
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "record_frames")

        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 720
        camera_properties.height = 480
        camera_handle = self.gym.create_camera_sensor(
            self.envs[0], camera_properties)
        self.camera_handle = camera_handle 

        self._setup_camera()
 
  # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        self._initial_root_states = self._root_states.clone()
        self._initial_root_states[:, 7:13] = 0

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)

        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self._rigid_body_rot = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]
        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, self.num_bodies, 3)
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
   
   
 
    def _create_sim(self):
        """创建仿真"""
        sim_params = gymapi.SimParams()
        sim_params.dt = self.sim_dt
        sim_params.num_client_threads = 0 
        sim_params.use_gpu_pipeline = True
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        # 使用 PhysX
        sim_params.physx.num_threads = 4
        sim_params.physx.solver_type = 1
        sim_params.physx.use_gpu = True 
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.contact_offset= 0.02
        sim_params.physx.rest_offset= 0.0
        sim_params.physx.bounce_threshold_velocity= 0.2
        sim_params.physx.max_depenetration_velocity= 10.0 
        sim_params.physx.default_buffer_size_multiplier= 5.0  
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024 
        sim_params.physx.num_subscenes= 4
        sim_params.physx.contact_collection= gymapi.ContactCollection(2)  # 0: CC_NEVER (don't collect contact info), 1: CC_LAST_SUBSTEP (collect only contacts on last substep), 2: CC_ALL_SUBSTEPS (broken - do not use!)

        self.gym = gymapi.acquire_gym()
        self.up_axis_idx = 2 # z

        self.sim = self.gym.create_sim(compute_device=0, graphics_device=0,
                                       type=gymapi.SIM_PHYSX, params=sim_params)

        if self.sim is None:
            raise Exception("Failed to create sim")
 
    def _create_ground_plane(self):
        """创建地面"""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)


    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids
    
    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id) 
        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids
    
    def _create_envs(self):
        """创建环境和机器人实例"""
        spacing = 5.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01 ##
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, self.asset_root, 
                                             self.asset_file, asset_options)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # create force sensors at the feet
        left_foot_name =  'left_ankle_roll_link'
        right_foot_name = 'right_ankle_roll_link'
        
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, left_foot_name)
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, right_foot_name)
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.8, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            contact_filter = 0
            
            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, contact_filter, 0)

            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            motor_efforts_np = self.motor_efforts.cpu().numpy()  
            dof_prop['effort'][:] = motor_efforts_np  
            self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        print("Stiffness (Kp):", dof_prop['stiffness'])
        print("Damping (Kd):", dof_prop['damping'])
        print("Drive Mode:", dof_prop['driveMode'])  # 应为 gymapi.DOF_MODE_POS
         
        self.p_gains = to_torch(dof_prop['stiffness'])
        self.d_gains = to_torch(dof_prop['damping'])
        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self._key_body_ids = self._build_key_body_ids_tensor(env_ptr, handle)
        self._contact_bodies = ["left_hand_link","right_hand_link","right_ankle_roll_link",
                                 "left_ankle_roll_link"]
 
        self._contact_body_ids = self._build_contact_body_ids_tensor(env_ptr, handle)
        
        self._pd_action_offset = np.array([-0.1,0.0,0.0,0.3,-0.2,0.0,-0.1,0.0,
                                      0.0,0.3,-0.2,0.0,0.0,0.0,0.0,0.0,
                                      0.2,0.0,1.1,0.0,-0.2,0.0,1.1])
        
        self._pd_action_scale = np.array([3.7873, 2.4435, 3.8606, 2.0769, 0.9774, 0.3665, 3.7873, 2.4435, 3.8606,
                                        2.0769, 0.9774, 0.3665, 3.6652, 0.7280, 0.7280, 4.0317, 2.6878, 3.6652,
                                        2.1991, 4.0317, 2.6878, 3.6652, 2.1991] )
 
        print(f"Created {self.num_envs} environments.")

    def _setup_camera(self):
        """ Set camera position and direction
        """
        position = [2, 0, 2]  # [m] # 相机的位置坐标 [x, y, z]
        lookat = [0., 0, 0.]  # [m] # 相机的目标位置，即相机看向的点的坐标
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    
    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "record_frames" and evt.value > 0:
                    self.record_frames = not self.record_frames

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)
            
            self.enable_viewer_sync = True
            self.render_fps = -1
            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

                # it seems like in some cases sync_frame_time still results in higher-than-realtime framerate
                # this code will slow down the rendering to real time
                now = time.time()
                delta = now - self.last_frame_time
                if self.render_fps < 0:
                    # render at control frequency
                    render_dt = self.sim_dt * self.control_freq_inv  # render every control step
                else:
                    render_dt = 1.0 / self.render_fps

                if delta < render_dt:
                    time.sleep(render_dt - delta)

                self.last_frame_time = time.time()

            else:
                self.gym.poll_viewer_events(self.viewer)
            
    def run(self):
        """运行仿真主循环""" 
   
        reset_time = 0.5
        sim_time = 0
       
        data = np.load(self.meanStdPath)
        mean_obs = data['mean'].astype(np.float32)   # (77,)
        std_obs = data['std'].astype(np.float32)     # (77,) 
        mean_obs = torch.tensor(mean_obs, dtype=torch.float32, device=self.device)
        std_obs = torch.tensor(std_obs, dtype=torch.float32, device=self.device)
 
        policy = onnxruntime.InferenceSession(self.onnxPath) 
        count_rl =0 
 
        while not self.gym.query_viewer_has_closed(self.viewer):
            # ================================
            # 设置所有关节力矩为 0
            # ================================
    
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            command = to_torch(np.array([x_vel_cmd,y_vel_cmd,yaw_vel_cmd ]))
            _curr_amp_obs_buf = build_observations(self._root_states,
                                                        self._dof_pos, 
                                                        self._dof_vel,
                                                        key_body_pos,
                                                        False,
                                                        command)
            normalized_obs = (_curr_amp_obs_buf - mean_obs) /  std_obs 
            cl_obs = torch.clamp(normalized_obs, -5, 5)
            policy_input = {policy.get_inputs()[0].name: cl_obs.cpu().numpy()}
            action = policy.run(["output"], policy_input)[0]

            # action = np.clip(action, -100, 100)
            target_dof_pos = action * self._pd_action_scale + self._pd_action_offset

            count_rl += 1 
            pd_tar = torch.tensor(target_dof_pos, dtype=torch.float32, device=self.device) 
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
            # ================================
            # 步进仿真
            # ================================
            for i in range(self.control_freq_inv):
                # 更新渲染
                self.render() 
                self.gym.simulate(self.sim)
    
                
                self.gym.refresh_dof_state_tensor(self.sim)
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_rigid_body_state_tensor(self.sim)

                self.gym.refresh_force_sensor_tensor(self.sim)
                self.gym.refresh_dof_force_tensor(self.sim)
                self.gym.refresh_net_contact_force_tensor(self.sim)

                # torques = self.p_gains * ( pd_tar - self._dof_pos ) - self.d_gains * self._dof_vel
                # print(torques)
                # print(self.dof_force_tensor ) 

            if self.device == 'cpu':    
                    self.gym.fetch_results(self.sim, True) 

            sim_time += self.sim_dt * self.control_freq_inv

        # 清理
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

# ================================
# 主函数
# ================================
if __name__ == "__main__":
    # 检查是否能找到 Isaac Gym
    print("Initializing Humanoid Simulation with Isaac Gym...")
    # 启动仿真
    sim = HumanoidSim()
    sim.run()