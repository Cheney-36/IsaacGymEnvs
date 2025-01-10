# Copyright (c) 2021-2023, NVIDIA Corporation
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

import numpy as np
import os


from isaacgym import gymtorch,gymapi,gymutil
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp, tf_combine, tf_vector,get_axis_params,quat_apply
from isaacgymenvs.tasks.base.vec_task import VecTask

import torch

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat

class InoPickPlace(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        print(" ==========================   ",self.cfg["name"])
        # 任务参数
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.num_props = self.cfg["env"]["numProps"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.grasp_reward_scale = self.cfg["env"]["graspRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1 / 60.

        # 机械臂和目标物体的尺寸
        self.prop_width = 0.08
        self.prop_height = 0.08
        self.prop_length = 0.08
        self.prop_spacing = 0.09

        num_obs = 23
        num_acts = 9

        self.cfg["env"]["numObservations"] = 23
        self.cfg["env"]["numActions"] = 9

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # 获取Gym GPU状态张量
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 创建一些包装张量
        self.robot_default_dof_pos = to_torch([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.robot_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_robot_dofs]
        self.robot_dof_pos = self.robot_dof_state[..., 0]
        self.robot_dof_vel = self.robot_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        if self.num_props > 0:
            self.prop_states = self.root_state_tensor[:, 2:]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.robot_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * (2 + self.num_props), dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        # robot_asset_file = "urdf/ino_ir_400_4/urdf/ino_ir_400_4.urdf"
        # stickers_asset_file = "urdf/stickers/urdf/stickers.urdf"

        # if "asset" in self.cfg["env"]:
        #     asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
        #     robot_asset_file = self.cfg["env"]["asset"].get("assetFileNameRobot", robot_asset_file)
        #     stickers_asset_file = self.cfg["env"]["asset"].get("assetFileNameStickers", stickers_asset_file)
        

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        robot_asset_file = "urdf/ino_ir_400_4/urdf/ino_ir_400_4.urdf"
        stickers_asset_file = "urdf/stickers/urdf/stickers.urdf"

        # if "asset" in self.cfg["env"]:
        #     asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
        #     robot_asset_file = self.cfg["env"]["asset"].get("assetFileNameRobot", robot_asset_file)
        #     stickers_asset_file = self.cfg["env"]["asset"].get("assetFileNameStickers", stickers_asset_file)

        # 加载机械臂资产
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)

        # 加载目标物体资产
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        stickers_asset = self.gym.load_asset(self.sim, asset_root, stickers_asset_file, asset_options)

        robot_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        robot_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)

        print("num robot bodies: ", self.num_robot_bodies)
        print("num robot dofs: ", self.num_robot_dofs)

        # 设置机械臂的DOF属性
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        for i in range(self.num_robot_dofs):
            robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                robot_dof_props['stiffness'][i] = robot_dof_stiffness[i]
                robot_dof_props['damping'][i] = robot_dof_damping[i]
            else:
                robot_dof_props['stiffness'][i] = 7000.0
                robot_dof_props['damping'][i] = 50.0

            self.robot_dof_lower_limits.append(robot_dof_props['lower'][i])
            self.robot_dof_upper_limits.append(robot_dof_props['upper'][i])

        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits, device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[[6, 7]] = 0.1
        robot_dof_props['effort'][6] = 200
        robot_dof_props['effort'][7] = 200

        # 创建目标物体资产
        box_opts = gymapi.AssetOptions()
        box_opts.density = 400
        prop_asset = self.gym.create_box(self.sim, self.prop_width, self.prop_height, self.prop_width, box_opts)

        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        stickers_start_pose = gymapi.Transform()
        stickers_start_pose.p = gymapi.Vec3(*get_axis_params(0.4, self.up_axis_idx))

        # 计算聚合大小
        num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        num_stickers_bodies = self.gym.get_asset_rigid_body_count(stickers_asset)
        num_stickers_shapes = self.gym.get_asset_rigid_shape_count(stickers_asset)
        num_prop_bodies = self.gym.get_asset_rigid_body_count(prop_asset)
        num_prop_shapes = self.gym.get_asset_rigid_shape_count(prop_asset)
        max_agg_bodies = num_robot_bodies + num_stickers_bodies + self.num_props * num_prop_bodies
        max_agg_shapes = num_robot_shapes + num_stickers_shapes + self.num_props * num_prop_shapes

        self.robots = []
        self.stickers = []
        self.default_prop_states = []
        self.prop_start = []
        self.envs = []

        for i in range(self.num_envs):
            # 创建环境实例
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            robot_actor = self.gym.create_actor(env_ptr, robot_asset, robot_start_pose, "robot", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            stickers_pose = stickers_start_pose
            stickers_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            dz = 0.5 * np.random.rand()
            dy = np.random.rand() - 0.5
            stickers_pose.p.y += self.start_position_noise * dy
            stickers_pose.p.z += self.start_position_noise * dz
            stickers_actor = self.gym.create_actor(env_ptr, stickers_asset, stickers_pose, "stickers", i, 2, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.num_props > 0:
                self.prop_start.append(self.gym.get_sim_actor_count(self.sim))
                drawer_handle = self.gym.find_actor_rigid_body_handle(env_ptr, stickers_actor, "drawer_top")
                drawer_pose = self.gym.get_rigid_transform(env_ptr, drawer_handle)

                props_per_row = int(np.ceil(np.sqrt(self.num_props)))
                xmin = -0.5 * self.prop_spacing * (props_per_row - 1)
                yzmin = -0.5 * self.prop_spacing * (props_per_row - 1)

                prop_count = 0
                for j in range(props_per_row):
                    prop_up = yzmin + j * self.prop_spacing
                    for k in range(props_per_row):
                        if prop_count >= self.num_props:
                            break
                        propx = xmin + k * self.prop_spacing
                        prop_state_pose = gymapi.Transform()
                        prop_state_pose.p.x = drawer_pose.p.x + propx
                        propz, propy = 0, prop_up
                        prop_state_pose.p.y = drawer_pose.p.y + propy
                        prop_state_pose.p.z = drawer_pose.p.z + propz
                        prop_state_pose.r = gymapi.Quat(0, 0, 0, 1)
                        prop_handle = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        prop_count += 1

                        prop_idx = j * props_per_row + k
                        self.default_prop_states.append([prop_state_pose.p.x, prop_state_pose.p.y, prop_state_pose.p.z,
                                                         prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z, prop_state_pose.r.w,
                                                         0, 0, 0, 0, 0, 0])
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.robots.append(robot_actor)
            self.stickers.append(stickers_actor)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, "hand_base_link")
        self.stickers_handle = self.gym.find_actor_rigid_body_handle(env_ptr, stickers_actor, "stickers")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, "leftfinger_tip")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, "rightfinger_tip")
        self.default_prop_states = to_torch(self.default_prop_states, device=self.device, dtype=torch.float).view(self.num_envs, self.num_props, 13)

        self.init_data()

    def init_data(self):
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robots[0], "hand_base_link")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robots[0], "leftfinger_tip")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robots[0], "rightfinger_tip")

        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

        finger_pose = gymapi.Transform()
        finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
        finger_pose.r = lfinger_pose.r

        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 1
        robot_local_grasp_pose = hand_pose_inv * finger_pose
        robot_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis))
        self.robot_local_grasp_pos = to_torch([robot_local_grasp_pose.p.x, robot_local_grasp_pose.p.y,
                                               robot_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = to_torch([robot_local_grasp_pose.r.x, robot_local_grasp_pose.r.y,
                                               robot_local_grasp_pose.r.z, robot_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        stickers_local_grasp_pose = gymapi.Transform()
        stickers_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.01, grasp_pose_axis, 0.3))
        stickers_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.stickers_local_grasp_pos = to_torch([stickers_local_grasp_pose.p.x, stickers_local_grasp_pose.p.y,
                                                  stickers_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.stickers_local_grasp_rot = to_torch([stickers_local_grasp_pose.r.x, stickers_local_grasp_pose.r.y,
                                                  stickers_local_grasp_pose.r.z, stickers_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.stickers_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.stickers_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))

        self.robot_grasp_pos = torch.zeros_like(self.robot_local_grasp_pos)
        self.robot_grasp_rot = torch.zeros_like(self.robot_local_grasp_rot)
        self.robot_grasp_rot[..., -1] = 1  # xyzw
        self.stickers_grasp_pos = torch.zeros_like(self.stickers_local_grasp_pos)
        self.stickers_grasp_rot = torch.zeros_like(self.stickers_local_grasp_rot)
        self.stickers_grasp_rot[..., -1] = 1
        self.robot_lfinger_pos = torch.zeros_like(self.robot_local_grasp_pos)
        self.robot_rfinger_pos = torch.zeros_like(self.robot_local_grasp_pos)
        self.robot_lfinger_rot = torch.zeros_like(self.robot_local_grasp_rot)
        self.robot_rfinger_rot = torch.zeros_like(self.robot_local_grasp_rot)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_robot_reward(
            self.reset_buf, self.progress_buf, self.actions, self.stickers_dof_pos,
            self.robot_grasp_pos, self.stickers_grasp_pos, self.robot_grasp_rot, self.stickers_grasp_rot,
            self.robot_lfinger_pos, self.robot_rfinger_pos,
            self.gripper_forward_axis, self.stickers_inward_axis, self.gripper_up_axis, self.stickers_up_axis,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.grasp_reward_scale,
            self.action_penalty_scale, self.distX_offset, self.max_episode_length
        )

    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        stickers_pos = self.rigid_body_states[:, self.stickers_handle][:, 0:3]
        stickers_rot = self.rigid_body_states[:, self.stickers_handle][:, 3:7]

        self.robot_grasp_rot[:], self.robot_grasp_pos[:], self.stickers_grasp_rot[:], self.stickers_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.robot_local_grasp_rot, self.robot_local_grasp_pos,
                                     stickers_rot, stickers_pos, self.stickers_local_grasp_rot, self.stickers_local_grasp_pos
                                     )

        self.robot_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.robot_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.robot_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.robot_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        dof_pos_scaled = (2.0 * (self.robot_dof_pos - self.robot_dof_lower_limits)
                          / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0)
        to_target = self.stickers_grasp_pos - self.robot_grasp_pos
        self.obs_buf = torch.cat((dof_pos_scaled, self.robot_dof_vel * self.dof_vel_scale, to_target,
                                  self.stickers_dof_pos[:, 3].unsqueeze(-1), self.stickers_dof_vel[:, 3].unsqueeze(-1)), dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # 重置机械臂
        pos = tensor_clamp(
            self.robot_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_robot_dofs), device=self.device) - 0.5),
            self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        self.robot_dof_pos[env_ids, :] = pos
        self.robot_dof_vel[env_ids, :] = torch.zeros_like(self.robot_dof_vel[env_ids])
        self.robot_dof_targets[env_ids, :self.num_robot_dofs] = pos

        # 重置目标物体
        self.stickers_dof_state[env_ids, :] = torch.zeros_like(self.stickers_dof_state[env_ids])

        # 重置道具
        if self.num_props > 0:
            prop_indices = self.global_indices[env_ids, 2:].flatten()
            self.prop_states[env_ids] = self.default_prop_states[env_ids]
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(prop_indices), len(prop_indices))

        multi_env_ids_int32 = self.global_indices[env_ids, :2].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.robot_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.robot_dof_targets[:, :self.num_robot_dofs] + self.robot_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.robot_dof_targets[:, :self.num_robot_dofs] = tensor_clamp(
            targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.robot_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # 调试可视化
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                px = (self.robot_grasp_pos[i] + quat_apply(self.robot_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.robot_grasp_pos[i] + quat_apply(self.robot_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.robot_grasp_pos[i] + quat_apply(self.robot_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.robot_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                px = (self.stickers_grasp_pos[i] + quat_apply(self.stickers_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.stickers_grasp_pos[i] + quat_apply(self.stickers_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.stickers_grasp_pos[i] + quat_apply(self.stickers_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.stickers_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.robot_lfinger_pos[i] + quat_apply(self.robot_lfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.robot_lfinger_pos[i] + quat_apply(self.robot_lfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.robot_lfinger_pos[i] + quat_apply(self.robot_lfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.robot_lfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.robot_rfinger_pos[i] + quat_apply(self.robot_rfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.robot_rfinger_pos[i] + quat_apply(self.robot_rfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.robot_rfinger_pos[i] + quat_apply(self.robot_rfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.robot_rfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_robot_reward(
    reset_buf, progress_buf, actions, stickers_dof_pos,
    robot_grasp_pos, stickers_grasp_pos, robot_grasp_rot, stickers_grasp_rot,
    robot_lfinger_pos, robot_rfinger_pos,
    gripper_forward_axis, stickers_inward_axis, gripper_up_axis, stickers_up_axis,
    num_envs, dist_reward_scale, rot_reward_scale, grasp_reward_scale,
    action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    # 从手到目标物体的距离
    d = torch.norm(robot_grasp_pos - stickers_grasp_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    axis1 = tf_vector(robot_grasp_rot, gripper_forward_axis)
    axis2 = tf_vector(stickers_grasp_rot, stickers_inward_axis)
    axis3 = tf_vector(robot_grasp_rot, gripper_up_axis)
    axis4 = tf_vector(stickers_grasp_rot, stickers_up_axis)

    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # 机械臂前向轴对齐
    dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # 机械臂上向轴对齐
    # 奖励机械臂与目标物体的方向匹配（手指包裹）
    rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

    # 如果左手指在目标物体上方，右手指在下方，则奖励
    grasp_reward = torch.zeros_like(rot_reward)
    grasp_reward = torch.where(robot_lfinger_pos[:, 2] > stickers_grasp_pos[:, 2],
                               torch.where(robot_rfinger_pos[:, 2] < stickers_grasp_pos[:, 2],
                                           grasp_reward + 0.5, grasp_reward), grasp_reward)

    # 动作正则化（每个环境的总和）
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # 奖励机械臂抓取目标物体
    grasp_reward = torch.where(d <= 0.02, grasp_reward + 1.0, grasp_reward)

    rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward \
        + grasp_reward_scale * grasp_reward - action_penalty_scale * action_penalty

    # 如果目标物体被抓取，则重置环境
    reset_buf = torch.where(d <= 0.02, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf

@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, robot_local_grasp_rot, robot_local_grasp_pos,
                             stickers_rot, stickers_pos, stickers_local_grasp_rot, stickers_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    global_robot_rot, global_robot_pos = tf_combine(
        hand_rot, hand_pos, robot_local_grasp_rot, robot_local_grasp_pos)
    global_stickers_rot, global_stickers_pos = tf_combine(
        stickers_rot, stickers_pos, stickers_local_grasp_rot, stickers_local_grasp_pos)

    return global_robot_rot, global_robot_pos, global_stickers_rot, global_stickers_pos