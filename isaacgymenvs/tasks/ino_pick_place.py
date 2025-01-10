
import numpy as np
import os

from isaacgym import gymtorch,gymapi, gymutil
from isaacgymenvs.utils.torch_jit_utils import to_torch, quat_apply, quat_conjugate, quat_mul, quat_from_angle_axis, tensor_clamp
from isaacgymenvs.tasks.base.vec_task import VecTask

import torch
from torch import Tensor
from typing import Dict
from typing import Tuple

class InoPickPlace(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.ino_position_noise = self.cfg["env"]["inoPositionNoise"]
        self.ino_rotation_noise = self.cfg["env"]["inoRotationNoise"]
        self.ino_dof_noise = self.cfg["env"]["inoDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # self.reward_settings = {
        #     "r_reach_scale": self.cfg["env"]["reachRewardScale"],
        #     "r_align_scale": self.cfg["env"]["alignRewardScale"],
        #     "r_pick_scale": self.cfg["env"]["pickRewardScale"],
        #     "r_place_scale": self.cfg["env"]["placeRewardScale"],
        #     "r_return_scale": self.cfg["env"]["returnRewardScale"],
        # }
        self.reward_settings = {
            "r_reach_scale": 1,
            "r_align_scale": 1,
            "r_pick_scale": 1,
            "r_place_scale": 1,
            "r_return_scale": 1,
        }

        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"joint_pos", "joint_tor"}, \
            "Invalid control type specified. Must be one of: {joint_pos, joint_tor}"

        self.cfg["env"]["numObservations"] = 23  # q(6) + eef_pose(7) + target_pose(7)
        self.cfg["env"]["numActions"] = 7  # q(6) + suction(1)

        self.states = {}
        self.handles = {}
        self.num_dofs = None
        self.actions = None
        self._init_target_state = None
        self._init_robot_state = None
        self._target_state = None
        self._robot_state = None
        self._root_state = None
        self._dof_state = None
        self._q = None
        self._qd = None
        self._rigid_body_state = None
        self._eef_state = None
        self._mm = None
        self._global_indices = None

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.ino_default_dof_pos = to_torch([0.0, -0.785, 0.0, -1.571, 0.0, 0.0], device=self.device)
        self.ino_dof_lower_limits = to_torch([-2.9, -1.7, -2.9, -2.5, -0.1, -2.9], device=self.device)
        self.ino_dof_upper_limits = to_torch([2.9, 1.7, 2.9, 2.5, 4.0, 2.9], device=self.device)
        self._ino_effort_limits = to_torch([87.0, 87.0, 87.0, 87.0, 12.0, 87.0], device=self.device)
        self.cmd_limit = self._ino_effort_limits.unsqueeze(0)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        robot_asset_file = "urdf/ino_ir_400_4/urdf/ino_ir_400_4_suck.urdf"
        stickers_asset_file = "urdf/stickers/urdf/stickers_suck.urdf"

        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, gymapi.AssetOptions())
        stickers_asset = self.gym.load_asset(self.sim, asset_root, stickers_asset_file, gymapi.AssetOptions())

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)

        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        robot_dof_props['driveMode'] = gymapi.DOF_MODE_POS
        robot_dof_props['stiffness'] = 400.0
        robot_dof_props['damping'] = 40.0

        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        stickers_start_pose = gymapi.Transform()
        stickers_start_pose.p = gymapi.Vec3(0.5, 0.0, 0.5)
        stickers_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        max_agg_bodies = self.num_robot_bodies + 1
        max_agg_shapes = self.gym.get_asset_rigid_shape_count(robot_asset) + self.gym.get_asset_rigid_shape_count(stickers_asset)

        self.robots = []
        self.envs = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            robot_actor = self.gym.create_actor(env_ptr, robot_asset, robot_start_pose, "robot", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)

            stickers_actor = self.gym.create_actor(env_ptr, stickers_asset, stickers_start_pose, "stickers", i, 2, 0)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.robots.append(robot_actor)

        self._init_robot_state = torch.zeros(self.num_envs, self.num_robot_dofs, device=self.device)
        self._init_target_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._target_state = self._root_state[:, 1, :]

        self.init_data()

    def init_data(self):
        env_ptr = self.envs[0]
        robot_handle = 0
        self.handles = {
            "eef": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "ee_link"),
            "stickers_body": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "stickers"),
        }

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["eef"], :]
        self._target_state = self._root_state[:, 1, :]
        self._mm = self.gym.acquire_mass_matrix_tensor(self.sim, "robot")

        self.states.update({
            "q": self._q[:, :6],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "target_pos": self._target_state[:, :3],
            "target_quat": self._target_state[:, 3:7],
        })

        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)
        self._arm_control = self._pos_control[:, :6]
        self._suction_control = self._pos_control[:, 6]

        self._global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            "q": self._q[:, :6],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "target_pos": self._target_state[:, :3],
            "target_quat": self._target_state[:, 3:7],
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_ino_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, self.max_episode_length
        )

    def compute_observations(self):
        self._refresh()
        obs = ["q", "eef_pos", "eef_quat", "target_pos", "target_quat"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        pos_noise = torch.rand((len(env_ids), self.num_robot_dofs), device=self.device) * 2 - 1
        pos = self.ino_default_dof_pos.unsqueeze(0) + pos_noise * self.ino_dof_noise
        pos[:, -1] = 0.0  # suction closed

        self._q[env_ids, :6] = pos[:, :6]
        self._qd[env_ids, :6] = torch.zeros_like(self._qd[env_ids, :6])

        self._pos_control[env_ids, :6] = pos[:, :6]
        self._effort_control[env_ids, :6] = torch.zeros_like(pos[:, :6])

        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._pos_control), gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._effort_control), gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state), gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        u_arm, u_suction = self.actions[:, :-1], self.actions[:, -1]

        u_arm_scaled = u_arm * self.cmd_limit / self.action_scale
        self._arm_control[:, :] = u_arm_scaled
        self._suction_control[:,:] = torch.where(u_suction >= 0.0, 1.0, 0.0)

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            eef_pos = self.states["eef_pos"]
            target_pos = self.states["target_pos"]
            for i in range(self.num_envs):
                self.gym.add_lines(self.viewer, self.envs[i], 1, [eef_pos[i].cpu().numpy(), target_pos[i].cpu().numpy()], [0.0, 1.0, 0.0])



# @torch.jit.script
# def compute_ino_reward(
#     reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
# ):
#     # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

#     target_pos = states["target_pos"]
#     eef_pos = states["eef_pos"]
#     eef_quat = states["eef_quat"]
#     q = states["q"]

#     reach_reward = 0.0
#     align_reward = 0.0
#     pick_reward = 0.0
#     place_reward = 0.0
#     return_reward = 0.0

#     # Reward for reaching point b
#     reach_dist = torch.norm(eef_pos - target_pos, dim=-1)
#     reach_reward = 1.0 - torch.tanh(10.0 * reach_dist)

#     # Reward for aligning with the target object
#     align_dist = torch.norm(quat_mul(eef_quat, quat_conjugate(states["target_quat"])), dim=-1)
#     align_reward = 1.0 - torch.tanh(10.0 * align_dist)

#     # Reward for picking up the object
#     pick_reward = torch.where(reach_dist < 0.1 and align_dist < 0.1, 1.0, 0.0)
#     point_d = [0,0,0,1,0,0,0]
#     # Reward for placing the object at point d
#     place_reward = torch.where(torch.norm(eef_pos - point_d, dim=-1) < 0.1, 1.0, 0.0)
    
#     ino_default_dof_pos = [0,0,0,1,0,0,0]
#     # Reward for returning to origin
#     return_reward = torch.where(torch.norm(q - ino_default_dof_pos, dim=-1) < 0.1, 1.0, 0.0)

#     # Compose rewards
#     rewards = reward_settings["r_reach_scale"] * reach_reward \
#             + reward_settings["r_align_scale"] * align_reward \
#             + reward_settings["r_pick_scale"] * pick_reward \
#             + reward_settings["r_place_scale"] * place_reward \
#             + reward_settings["r_return_scale"] * return_reward

#     # Compute resets
#     reset_buf = torch.where((progress_buf >= max_episode_length - 1) \
#                             or (return_reward > 0), torch.ones_like(reset_buf), reset_buf)

#     return rewards, reset_buf

@torch.jit.script
def compute_ino_reward(
    reset_buf: Tensor, progress_buf: Tensor, actions: Tensor, states: Dict[str, Tensor], 
    reward_settings: Dict[str, float], max_episode_length: float
) -> Tuple[Tensor, Tensor]:

    target_pos: Tensor = states["target_pos"]
    eef_pos: Tensor = states["eef_pos"]
    eef_quat: Tensor = states["eef_quat"]
    q: Tensor = states["q"]

    # Define constants
    point_d_pos: Tensor = torch.tensor([0.0, 0.0, 0.0], device=eef_pos.device, dtype=eef_pos.dtype)
    ino_default_dof_pos: Tensor = torch.tensor([0.0, -0.785, 0.0, -1.571, 0.0, 0.0], device=q.device, dtype=q.dtype)

    # Reward components
    reach_dist: Tensor = torch.norm(eef_pos - target_pos, dim=-1)
    reach_reward: Tensor = 1.0 - torch.tanh(10.0 * reach_dist)

    align_dist: Tensor = torch.norm(quat_mul(eef_quat, quat_conjugate(states["target_quat"])), dim=-1)
    align_reward: Tensor = 1.0 - torch.tanh(10.0 * align_dist)

    pick_reward: Tensor = torch.where(torch.logical_and(reach_dist < 0.1, align_dist < 0.1), 1.0, 0.0)

    place_dist: Tensor = torch.norm(eef_pos - point_d_pos, dim=-1)
    place_reward: Tensor = torch.where(place_dist < 0.1, 1.0, 0.0)

    return_dist: Tensor = torch.norm(q - ino_default_dof_pos, dim=-1)
    return_reward: Tensor = torch.where(return_dist < 0.1, 1.0, 0.0)

    # Compose rewards
    rewards: Tensor = reward_settings["r_reach_scale"] * reach_reward \
                     + reward_settings["r_align_scale"] * align_reward \
                     + reward_settings["r_pick_scale"] * pick_reward \
                     + reward_settings["r_place_scale"] * place_reward \
                     + reward_settings["r_return_scale"] * return_reward

    # Compute resets
    reset_condition: Tensor = (progress_buf >= max_episode_length - 1) | (return_reward > 0.0)
    reset_buf: Tensor = torch.where(reset_condition, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf