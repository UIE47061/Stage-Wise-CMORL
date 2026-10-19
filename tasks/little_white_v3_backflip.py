# Copyright (c) 2018-2022, NVIDIA Corporation
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

from isaacgym import torch_utils
from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.tasks.base.vec_task import VecTask

import numpy as np
import torch
import os

class Env(VecTask):
    def __init__(self,
                 cfg: dict,
                 rl_device: str,
                 sim_device: str,
                 graphics_device_id: int,
                 headless: bool,
                 virtual_screen_capture: bool,
                 force_render: bool,
                 ):

        """
        raw_obs:
            base orn (3), joint pos (12), joint vel (12), prev action (12), command (3)
        observation: raw_obs * history_len
        state:
            linear velocity (3), angular vel (3), com height (1), foot contact (4),
            gravity (3), friction (1), restitution (1), stage (5)
        """
        self.cfg = cfg
        self.motor_cfg = cfg['env']['control'].get('motor_model', {})
        self.use_motor_model = self.motor_cfg.get('enabled', False)
        self.motor_no_load_speed = float(self.motor_cfg.get('no_load_speed_rad_s', 0))
        if self.use_motor_model and (not np.isfinite(self.motor_no_load_speed) or self.motor_no_load_speed <= 0):
            raise ValueError('Enabled motor model requires a finite positive no-load speed.')
        if self.use_motor_model and self.motor_cfg.get('model') != 'linear_motoring_envelope':
            raise ValueError('Unsupported motor model; expected linear_motoring_envelope.')
        if self.use_motor_model:
            print(f'[little_white] Approximate motor envelope enabled: no-load {self.motor_no_load_speed:.3f} rad/s; '
                  'hardware version/voltage and braking/thermal behavior still require validation.', flush=True)
        backflip_cfg = self.cfg["env"]["backflip"]
        self.go1_transitions = backflip_cfg.get('transition_mode', 'legacy') == 'go1'
        if backflip_cfg.get('transition_mode', 'legacy') not in ('legacy', 'go1'):
            raise ValueError('transition_mode must be legacy or go1')
        self.launch_cfg = backflip_cfg.get('launch_training', {})
        self.flight_cfg = backflip_cfg.get('flight_training', {})
        self.landing_cfg = backflip_cfg.get('landing_training', {})
        self.preparation_cfg = backflip_cfg.get('preparation_training', {})
        self.use_preparation_training = self.preparation_cfg.get('enabled', False)
        if self.use_preparation_training:
            for key in ('smooth_weight', 'angular_weight', 'vertical_weight', 'support_weight', 'calf_weight'):
                value = self.preparation_cfg[key]
                if not np.isfinite(value) or value < 0:
                    raise ValueError('Preparation reward weights must be finite and nonnegative.')
        self.use_landing_training = self.landing_cfg.get('enabled', False)
        if self.use_landing_training:
            if not (self.go1_transitions or backflip_cfg.get('strict_flight_validation', False)):
                raise ValueError('Landing training requires independent strict success evaluation.')
            for key in ('support_weight', 'success_bonus'):
                value = self.landing_cfg[key]
                if not np.isfinite(value) or value < 0:
                    raise ValueError('Landing reward weights must be finite and nonnegative.')
        self.use_flight_training = self.flight_cfg.get('enabled', False)
        if self.use_flight_training:
            if not backflip_cfg.get('strict_flight_validation', False):
                raise ValueError('Flight training requires strict flight validation.')
            for key in ('progress_weight', 'launch_rotation_weight'):
                if not np.isfinite(self.flight_cfg[key]) or self.flight_cfg[key] <= 0:
                    raise ValueError(f'flight_training.{key} must be finite and positive.')
        self.use_launch_training = self.launch_cfg.get('enabled', False)
        if self.use_launch_training:
            for key in ('jump_timeout_s', 'command_timeout_s', 'takeoff_bonus', 'timeout_penalty'):
                if not np.isfinite(self.launch_cfg[key]) or self.launch_cfg[key] <= 0:
                    raise ValueError(f'launch_training.{key} must be finite and positive.')
            if not backflip_cfg.get('strict_flight_validation', False):
                raise ValueError('Launch training requires strict flight validation.')
        self.stand_height = backflip_cfg["stand_height"]
        self.sit_height = backflip_cfg["sit_height"]
        self.jump_reward_height_cap = backflip_cfg["jump_reward_height_cap"]
        self.stand_transition_height = backflip_cfg["stand_transition_height"]
        self.sit_transition_height = backflip_cfg["sit_transition_height"]
        self.takeoff_confirm_steps = int(backflip_cfg["takeoff_confirm_steps"])
        self.min_air_time = backflip_cfg["min_air_time"]
        self.landing_min_downward_velocity = backflip_cfg["landing_min_downward_velocity"]
        self.landing_upright_cos = backflip_cfg["landing_upright_cos"]
        self.landing_stable_time = backflip_cfg["landing_stable_time"]
        self.landing_max_vertical_speed = backflip_cfg["landing_max_vertical_speed"]
        self.landing_max_angular_speed = backflip_cfg["landing_max_angular_speed"]
        self.landing_max_horizontal_speed = backflip_cfg.get('landing_max_horizontal_speed', 0.30)
        self.pose_cfg = backflip_cfg.get('pose_guidance', {})
        self.use_pose_guidance = self.pose_cfg.get('enabled', False)
        self.strict_flight_validation = backflip_cfg.get('strict_flight_validation', self.use_pose_guidance)
        if self.go1_transitions and self.strict_flight_validation:
            raise ValueError('Go1 training transitions require strict_flight_validation=false; evaluation remains strict.')
        self.early_tuck_cfg = backflip_cfg.get('early_tuck', {})
        self.use_early_tuck = self.early_tuck_cfg.get('enabled', False)
        if self.use_early_tuck:
            if not (self.strict_flight_validation and self.use_pose_guidance):
                raise ValueError('Early tuck requires pose guidance and strict flight validation.')
            ramp = self.early_tuck_cfg.get('ramp_steps', 2)
            if isinstance(ramp, bool) or not isinstance(ramp, int) or ramp < 1:
                raise ValueError('early_tuck.ramp_steps must be a positive integer.')
            if not isinstance(self.early_tuck_cfg.get('blend_pd', True), bool):
                raise ValueError('early_tuck.blend_pd must be a boolean.')
        self.minimum_air_rotation = float(backflip_cfg.get('minimum_air_rotation', 4.5))
        if not 0 < self.minimum_air_rotation <= 2*np.pi:
            raise ValueError('minimum_air_rotation must be in (0, 2*pi].')
        if self.use_pose_guidance:
            print('[little_white] EXPERIMENTAL simulation posture guidance; stable backflip not yet validated. '
                  'Joint velocity limits are soft costs, not hard speed clamps. See README_LITTLE_WHITE_V3_ZH.md.', flush=True)
        self.raw_obs_dim = 3 + 12*3 + 3
        self.history_len = self.cfg["env"]["history_len"]
        self.cfg['env']['numObservations'] = self.raw_obs_dim * self.history_len
        self.cfg['env']['numStates'] = 3 + 3 + 1 + 4 + 3 + 1 + 1 + 5
        self.sim_dt = self.cfg["sim"]["sim_dt"]
        self.control_dt = self.cfg["sim"]["con_dt"]
        self.landing_stable_steps = max(
            1, int(np.ceil(self.landing_stable_time / self.control_dt))
        )
        self.cfg["env"]["controlFrequencyInv"] = int(self.control_dt/self.sim_dt + 0.5)
        self.cfg['sim']['dt'] = self.sim_dt
        self.cfg['physics_engine'] = self.cfg['sim']['physics_engine']
        self.cfg['env']['enableCameraSensors'] = self.cfg['env']['enable_camera_sensors']
        self.cfg['env']['numEnvs'] = self.cfg['env']['num_envs']
        self.cfg['env']['numActions'] = 12
        self.num_legs = 4

        # for randomization
        self.is_randomized = self.cfg["env"]["randomize"]["is_randomized"]
        self.rand_period_motor_strength_s = self.cfg["env"]["randomize"]["rand_period_motor_strength_s"]
        self.rand_period_gravity_s = self.cfg["env"]["randomize"]["rand_period_gravity_s"]
        self.rand_period_motor_strength = int(self.rand_period_motor_strength_s/self.control_dt + 0.5)
        self.rand_period_gravity = int(self.rand_period_gravity_s/self.control_dt + 0.5)
        self.rand_range_body_mass = self.cfg["env"]["randomize"]["rand_range_body_mass"]
        self.rand_range_com_pos_x = self.cfg["env"]["randomize"]["rand_range_com_pos_x"]
        self.rand_range_com_pos_y = self.cfg["env"]["randomize"]["rand_range_com_pos_y"]
        self.rand_range_com_pos_z = self.cfg["env"]["randomize"]["rand_range_com_pos_z"]
        self.rand_range_dof_pos = self.cfg["env"]["randomize"]["rand_range_init_dof_pos"]
        self.rand_range_root_vel = self.cfg["env"]["randomize"]["rand_range_init_root_vel"]
        self.rand_range_motor_strength = self.cfg["env"]["randomize"]["rand_range_motor_strength"]
        self.rand_range_gravity = self.cfg["env"]["randomize"]["rand_range_gravity"]
        self.rand_range_friction = self.cfg["env"]["randomize"]["rand_range_friction"]
        self.rand_range_restitution = self.cfg["env"]["randomize"]["rand_range_restitution"]
        self.rand_range_motor_offset = self.cfg["env"]["randomize"]["rand_range_motor_offset"]
        self.noise_range_dof_pos = self.cfg["env"]["randomize"]["noise_range_dof_pos"]
        self.noise_range_dof_vel = self.cfg["env"]["randomize"]["noise_range_dof_vel"]
        self.noise_range_body_orn = self.cfg["env"]["randomize"]["noise_range_body_orn"]
        self.n_lag_action_steps = self.cfg["env"]["randomize"]["n_lag_action_steps"]
        self.n_lag_imu_steps = self.cfg["env"]["randomize"]["n_lag_imu_steps"]
        self.common_step_counter = 0

        """
        In the parent's __init__, generate the following member variables:
            - self.device_type: {cuda, gpu, cpu}
            - self.device_id: {0, 1, ...}
            - self.device: {cuda:0, cuda:1, ..., cpu}
            - self.rl_device
            - self.headless
            - self.graphics_device_id
            - self.num_environments: config['env']['numEnvs']
            - self.num_observations: config['env']['numObservations']
            - self.num_actions: config['env']['numActions']
            - self.control_freq_inv: {1, config['env']['controlFrequencyInv']}
            - self.obs_space, self.act_space
            - self.clip_obs, self.clip_actions
            - self.virtual_screen_capture, self.force_render
            - self.sim_params: self.__parse_sim_params(self.cfg["physics_engine"], self.cfg["sim"])
            - self.physics_engine: {gymapi.SIM_PHYSX, gymapi.SIM_FLEX}
        Then, called:
            - self.create_sim()
            - self.set_viewer()
            - self.allocate_buffers()
        """
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.max_episode_length_s = self.cfg["env"]["learn"]["episode_length_s"]
        self.max_episode_length = int(self.max_episode_length_s/self.control_dt + 0.5)
        self.reward_names = self.cfg["env"]["reward_names"]
        self.cost_names = self.cfg["env"]["cost_names"]
        self.stage_names = self.cfg["env"]["stage_names"]
        self.num_rewards = len(self.reward_names)
        self.num_costs = len(self.cost_names)
        self.num_stages = len(self.stage_names)
        self.action_smooth_weight = self.cfg["env"]["control"]["action_smooth_weight"]
        self.action_scale = self.cfg["env"]["control"]["action_scale"]

        # for buffer
        self.rew_buf = torch.zeros((self.num_envs, self.num_rewards),
                                    dtype=torch.float32, device=self.device)
        self.cost_buf = torch.zeros((self.num_envs, self.num_costs),
                                    dtype=torch.float32, device=self.device)
        self.stage_buf = torch.zeros((self.num_envs, self.num_stages),
                                    dtype=torch.float32, device=self.device)
        self.fail_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # check the robot tumbling
        self.is_half_turn_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.is_one_turn_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.verified_turn_buf = torch.zeros_like(self.is_one_turn_buf)
        self.start_time_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.cmd_time_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.land_time_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.takeoff_counter_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.air_time_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.landing_stable_counter_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.landing_success_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.flip_angle_buf = torch.zeros(self.num_envs, device=self.device)
        self.prev_pitch_buf = torch.zeros(self.num_envs, device=self.device)
        self.prev_vertical_velocity = torch.zeros(self.num_envs, device=self.device)
        self.last_contact_angle_buf = torch.zeros(self.num_envs, device=self.device)
        self.air_rotation_buf = torch.zeros(self.num_envs, device=self.device)
        self.true_air_rotation_buf = torch.zeros(self.num_envs, device=self.device)
        self.peak_true_air_rotation_buf = torch.zeros(self.num_envs, device=self.device)
        self.previous_free_air_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.early_tuck_weight_buf = torch.zeros(self.num_envs, device=self.device)
        self.jump_time_buf = torch.zeros(self.num_envs, device=self.device)

        # get gym state tensors
        """
        It can be considered as the following variables get GPU pointers.
        After refresh state tensors, value of the following variables are updated automatically.
        """
        raw_actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        raw_dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        raw_dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        raw_net_contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        raw_rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # update state and force tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(
                            raw_actor_root_state_tensor).view(self.num_envs, 13)
        self.dof_states = gymtorch.wrap_tensor(
                            raw_dof_state_tensor).view(self.num_envs, self.num_dofs, 2)
        self.dof_torques = gymtorch.wrap_tensor(
                            raw_dof_force_tensor).view(self.num_envs, self.num_dofs)
        self.contact_forces = gymtorch.wrap_tensor(
                            raw_net_contact_force_tensor).view(self.num_envs, self.num_bodies, 3)
        self.rigid_body_states = gymtorch.wrap_tensor(
                            raw_rigid_body_state_tensor).view(self.num_envs, self.num_bodies, 13)
        self.base_positions = self.root_states[:, :3]
        self.base_quaternions = self.root_states[:, 3:7]
        self.base_lin_vels = self.root_states[:, 7:10]
        self.base_ang_vels = self.root_states[:, 10:13]
        self.dof_positions = self.dof_states[..., 0]
        self.dof_velocities = self.dof_states[..., 1]

        # setting viewer
        if self.viewer != None:
            p = self.base_positions[0]
            cam_pos = gymapi.Vec3(p[0] + 5.0, p[1] + 5.0, p[2] + 3.0)
            cam_target = gymapi.Vec3(*p)
            self.gym.viewer_camera_look_at(
                self.viewer, self.env_handles[0], cam_pos, cam_target)

        # get default base frame pose
        base_init_state = []
        base_init_state += self.cfg["env"]["init_base_pose"]["pos"]
        base_init_state += self.cfg["env"]["init_base_pose"]["quat"]
        base_init_state += self.cfg["env"]["init_base_pose"]["lin_vel"]
        base_init_state += self.cfg["env"]["init_base_pose"]["ang_vel"]
        self.base_init_state = torch_utils.to_torch(
            base_init_state, dtype=torch.float32, device=self.device, requires_grad=False)

        # get default joint (DoF) position
        self.named_default_joint_positions = self.cfg["env"]["default_joint_positions"]
        self.default_dof_positions = torch.zeros_like(
            self.dof_positions, dtype=torch.float32, device=self.device, requires_grad=False)
        for i in range(self.num_actions):
            name = self.dof_names[i]
            self.default_dof_positions[:, i] = self.named_default_joint_positions[name]

        # Keep the policy's action origin unchanged, so existing actors can be
        # warm-started. These are stage-specific references, not new offsets.
        self.stage_pose_targets = self.default_dof_positions[0].repeat(self.num_stages, 1)
        self.rear_dof_indices = torch.tensor([i for i, name in enumerate(self.dof_names)
                                             if name.startswith(('RL_', 'RR_'))], device=self.device)
        if self.use_pose_guidance:
            for stage_idx, stage_name in enumerate(self.stage_names):
                pose = self.pose_cfg['poses'][stage_name]
                for joint_idx, name in enumerate(self.dof_names):
                    leg, joint, _ = name.split('_')
                    group = 'front' if leg in ('FL', 'FR') else 'rear'
                    self.stage_pose_targets[stage_idx, joint_idx] = pose[group][('Hip', 'Thigh', 'Calf').index(joint)]
            self.pose_blends = torch.tensor(self.pose_cfg['control_blend'], device=self.device)
            if self.pose_blends.shape != (self.num_stages,) or ((self.pose_blends < 0) | (self.pose_blends > 1)).any():
                raise ValueError('control_blend needs five values between 0 and 1.')
            if not torch.isfinite(self.pose_blends).all() or not torch.isfinite(self.stage_pose_targets).all():
                raise ValueError('Pose targets and blend weights must be finite.')
            if not (0 <= self.pose_cfg['untuck_start_angle'] < self.pose_cfg['untuck_end_angle']):
                raise ValueError('Untuck angles must be nonnegative and strictly increasing.')
            if self.pose_cfg.get('rear_push_delay', 0) > 0 and self.pose_cfg.get('rear_push_ramp', 0) <= 0:
                raise ValueError('Delayed rear-leg push needs a positive ramp duration.')
            if 'stiffness' in self.pose_cfg:
                self.stage_stiffness = torch.tensor(self.pose_cfg['stiffness'], device=self.device)
                self.stage_damping = torch.tensor(self.pose_cfg['damping'], device=self.device)
                if self.stage_stiffness.shape != (5,) or self.stage_damping.shape != (5,):
                    raise ValueError('Stage stiffness and damping need five entries.')
                if (self.stage_stiffness <= 0).any() or (self.stage_damping < 0).any():
                    raise ValueError('Invalid stage PD gains.')
                if not torch.isfinite(self.stage_stiffness).all() or not torch.isfinite(self.stage_damping).all():
                    raise ValueError('Stage PD gains must be finite.')

        # for inner variables
        self.world_x = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.world_y = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.world_z = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.world_x[:, 0] = 1.0
        self.world_y[:, 1] = 1.0
        self.world_z[:, 2] = 1.0
        self.joint_targets = torch.zeros(
            (self.num_envs, self.num_dofs),
            dtype=torch.float32, device=self.device, requires_grad=False)
        self.prev_actions = torch.zeros(
            (self.num_envs, self.num_dofs),
            dtype=torch.float32, device=self.device, requires_grad=False)
        self.motor_strengths = torch.ones((self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device, requires_grad=False)
        self.motor_offsets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device, requires_grad=False)
        self.lag_joint_target_buffer = [torch.zeros_like(self.dof_positions) for _ in range(self.n_lag_action_steps + 1)]
        self.lag_imu_buffer = [torch.zeros_like(self.world_z) for _ in range(self.n_lag_imu_steps + 1)]
        self.prev_joint_targets = torch.zeros_like(self.joint_targets)
        self.prev_prev_joint_targets = torch.zeros_like(self.joint_targets)
        self.gravity = torch.tensor(self.cfg['sim']['gravity'], dtype=torch.float32, device=self.device, requires_grad=False)

        # for dof limits
        dof_pos_lower_limits = []
        dof_pos_upper_limits = []
        for joint_name in ['hip', 'thigh', 'calf']:
            joint_dict = self.cfg["env"]["learn"][f"{joint_name}_joint_limit"]
            dof_pos_lower_limits.append(joint_dict['lower'] if 'lower' in joint_dict.keys() else -np.inf)
            dof_pos_upper_limits.append(joint_dict['upper'] if 'upper' in joint_dict.keys() else np.inf)
        self.dof_pos_lower_limits = torch_utils.to_torch(
            dof_pos_lower_limits*4, dtype=torch.float32, device=self.device, requires_grad=False)
        self.dof_pos_lower_limits = torch.maximum(self.dof_pos_lower_limits, self.default_dof_pos_lower_limits)
        self.dof_pos_upper_limits = torch_utils.to_torch(
            dof_pos_upper_limits*4, dtype=torch.float32, device=self.device, requires_grad=False)
        self.dof_pos_upper_limits = torch.minimum(self.dof_pos_upper_limits, self.default_dof_pos_upper_limits)
        if self.use_pose_guidance and ((self.stage_pose_targets < self.dof_pos_lower_limits) |
                                      (self.stage_pose_targets > self.dof_pos_upper_limits)).any():
            raise ValueError('Stage pose references must stay within configured joint limits.')
        self.dof_vel_upper_limits = torch_utils.to_torch(
            self.cfg["env"]["learn"]["joint_vel_upper"], dtype=torch.float32, device=self.device, requires_grad=False)
        self.dof_vel_upper_limits = torch.minimum(self.dof_vel_upper_limits, self.default_dof_vel_upper_limits)
        self.dof_torques_upper_limits = torch_utils.to_torch(
            self.cfg["env"]["learn"]["joint_torque_upper"], dtype=torch.float32, device=self.device, requires_grad=False)
        self.dof_torques_upper_limits = torch.minimum(self.dof_torques_upper_limits, self.default_dof_torques_upper_limits)

        # for noise observation
        self.est_base_body_orns = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.est_dof_positions = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device, requires_grad=False)
        self.est_dof_velocities = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device, requires_grad=False)

        # for observation and action symmetric matrix
        self.joint_sym_mat = torch.zeros((self.num_dofs, self.num_dofs), device=self.device, dtype=torch.float32, requires_grad=False)
        self.joint_sym_mat[:3, 3:6] = torch.eye(3, device=self.device, dtype=torch.float32)
        self.joint_sym_mat[0, 3] = -1.0
        self.joint_sym_mat[3:6, :3] = torch.eye(3, device=self.device, dtype=torch.float32)
        self.joint_sym_mat[3, 0] = -1.0
        self.joint_sym_mat[6:9, 9:12] = torch.eye(3, device=self.device, dtype=torch.float32)
        self.joint_sym_mat[6, 9] = -1.0
        self.joint_sym_mat[9:12, 6:9] = torch.eye(3, device=self.device, dtype=torch.float32)
        self.joint_sym_mat[9, 6] = -1.0
        self.obs_sym_mat = torch.zeros((self.num_obs, self.num_obs), device=self.device, dtype=torch.float32, requires_grad=False)
        raw_obs_sym_mat = torch.eye(self.raw_obs_dim, device=self.device, dtype=torch.float32, requires_grad=False)
        raw_obs_sym_mat[1, 1] = -1.0
        for i in range(3):
            raw_obs_sym_mat[(3+self.num_dofs*(i)):(3+self.num_dofs*(i+1)), (3+self.num_dofs*(i)):(3+self.num_dofs*(i+1))] = self.joint_sym_mat.clone()
        raw_obs_sym_mat[3+3*self.num_dofs:, 3+3*self.num_dofs:] = torch.eye(3, device=self.device, dtype=torch.float32)
        for i in range(self.history_len):
            self.obs_sym_mat[(self.raw_obs_dim*i):(self.raw_obs_dim*(i+1)), (self.raw_obs_dim*i):(self.raw_obs_dim*(i+1))] = raw_obs_sym_mat.clone()
        self.state_sym_mat = torch.eye(self.num_states - self.num_stages, device=self.device, dtype=torch.float32, requires_grad=False)
        self.state_sym_mat[1, 1] = -1.0
        self.state_sym_mat[3, 3] = -1.0
        self.state_sym_mat[5, 5] = -1.0
        self.state_sym_mat[7:11, 7:11] = 0
        self.state_sym_mat[7, 8] = 1.0
        self.state_sym_mat[8, 7] = 1.0
        self.state_sym_mat[9, 10] = 1.0
        self.state_sym_mat[10, 9] = 1.0
        self.state_sym_mat[12, 12] = -1.0


    def create_sim(self):
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        # make terrain
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

        # set asset location
        asset_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/assets'
        asset_file = self.cfg["env"]["urdf_asset"]["file"]
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # create actor (or robot)'s asset option
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = False
        # Keep the four fixed foot links so their contact forces can be
        # measured independently from the calf links.
        asset_options.collapse_fixed_joints = self.cfg['env']['urdf_asset']['collapse_fixed_joints']
        asset_options.fix_base_link = self.cfg["env"]["urdf_asset"]["fix_base_link"]
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = self.cfg['env']['urdf_asset']['flip_visual_attachments']
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 1000.0
        asset_options.max_linear_velocity = 1000.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01

        # load asset
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        if self.num_dofs != 12:
            raise ValueError(f"Little White V3 Backflip expects 12 DoFs, got {self.num_dofs}.")
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # set friction and restitution
        if self.is_randomized:
            self.friction_coeffs = torch_utils.torch_rand_float(
                self.rand_range_friction[0], self.rand_range_friction[1], (self.num_envs, 1), device=self.device)
            self.restitution_coeffs = torch_utils.torch_rand_float(
                self.rand_range_restitution[0], self.rand_range_restitution[1], (self.num_envs, 1), device=self.device)
        else:
            self.friction_coeffs = torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False)
            self.restitution_coeffs = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False)

        # get an set dof property
        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.default_dof_pos_lower_limits = torch_utils.to_torch(
            dof_props['lower'], dtype=torch.float32, device=self.device, requires_grad=False)
        self.default_dof_pos_upper_limits = torch_utils.to_torch(
            dof_props['upper'], dtype=torch.float32, device=self.device, requires_grad=False)
        self.default_dof_vel_upper_limits = torch_utils.to_torch(
            dof_props['velocity'], dtype=torch.float32, device=self.device, requires_grad=False)
        self.default_dof_torques_upper_limits = torch_utils.to_torch(
            dof_props['effort'], dtype=torch.float32, device=self.device, requires_grad=False)

        # create environment and load assets
        spacing = self.cfg['env']['env_spacing']
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        num_envs_per_row = int(np.sqrt(self.num_envs))
        start_pose = gymapi.Transform()
        # Spawn directly at the configured reset pose.  Spawning at z=0 first
        # makes the hip collision shapes penetrate the ground and leaves a
        # large contact impulse in the first physics frame, causing an
        # immediate reset loop.
        init_pos = self.cfg["env"]["init_base_pose"]["pos"]
        init_quat = self.cfg["env"]["init_base_pose"]["quat"]
        start_pose.p = gymapi.Vec3(*init_pos)
        start_pose.r = gymapi.Quat(*init_quat)
        self.robot_handles = []
        self.env_handles = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_envs_per_row)
            # set rigid shape props
            for s in range(len(rigid_shape_props_asset)):
                rigid_shape_props_asset[s].friction = self.friction_coeffs[i, 0]
                rigid_shape_props_asset[s].restitution = self.restitution_coeffs[i, 0]
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props_asset)
            # create robot instance
            # The CAD-derived hip collision cylinders overlap the base at the
            # joints.  Disable intra-robot collisions so these adjacent links
            # do not generate large false contact forces; ground contacts are
            # unaffected by this actor collision filter.
            collision_filter = 1 if self.cfg["env"]["urdf_asset"].get("disable_self_collisions", True) else 0
            robot_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, "robot", i, collision_filter, 0
            )
            # ========== randomize base frame's mass & CoM position ========== #
            if self.is_randomized:
                body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
                body_props[0].mass += (np.random.rand()\
                                    *(self.rand_range_body_mass[1] - self.rand_range_body_mass[0]) \
                                    + self.rand_range_body_mass[0])
                com_x = np.random.rand() * (self.rand_range_com_pos_x[1] - self.rand_range_com_pos_x[0]) + self.rand_range_com_pos_x[0]
                com_y = np.random.rand() * (self.rand_range_com_pos_y[1] - self.rand_range_com_pos_y[0]) + self.rand_range_com_pos_y[0]
                com_z = np.random.rand() * (self.rand_range_com_pos_z[1] - self.rand_range_com_pos_z[0]) + self.rand_range_com_pos_z[0]
                if self.cfg['env']['randomize'].get('com_offset_relative', False):
                    com_x += body_props[0].com.x
                    com_y += body_props[0].com.y
                    com_z += body_props[0].com.z
                body_props[0].com = gymapi.Vec3(com_x, com_y, com_z)
                self.gym.set_actor_rigid_body_properties(env_handle, robot_handle, body_props, recomputeInertia=True)
            # ================================================================ #
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, robot_handle)
            self.robot_handles.append(robot_handle)
            self.env_handles.append(env_handle)

        # find link & joint names.
        self.link_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # Keep a deterministic FR, FL, RL, RR order. Reward/cost code uses
        # indices 2 and 3 as the rear feet during the jump stage.
        leg_names = ['FR', 'FL', 'RL', 'RR']
        base_names = ['base_link']
        hip_names = [f'{leg}_Hip_link' for leg in leg_names]
        thigh_names = [f'{leg}_Thigh_link' for leg in leg_names]
        calf_names = [f'{leg}_Calf_link' for leg in leg_names]
        foot_names = [f'{leg}_Foot_link' for leg in leg_names]
        required_link_names = base_names + hip_names + thigh_names + calf_names + foot_names
        missing_link_names = [name for name in required_link_names if name not in self.link_names]
        if missing_link_names:
            raise ValueError(f"Little White V3 URDF is missing links: {missing_link_names}")
        terminate_touch_names = base_names + hip_names
        undesired_touch_names = thigh_names + calf_names

        # find foot & knee & hip & base's index
        self.foot_indices = torch.zeros(
            len(foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.calf_indices = torch.zeros(
            len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.terminate_touch_indices = torch.zeros(
            len(terminate_touch_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.undesired_touch_indices = torch.zeros(
            len(undesired_touch_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(foot_names)):
            self.foot_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.env_handles[0], self.robot_handles[0], foot_names[i])
        for i in range(len(calf_names)):
            self.calf_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.env_handles[0], self.robot_handles[0], calf_names[i])
        for i in range(len(terminate_touch_names)):
            self.terminate_touch_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.env_handles[0], self.robot_handles[0], terminate_touch_names[i])
        for i in range(len(undesired_touch_names)):
            self.undesired_touch_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.env_handles[0], self.robot_handles[0], undesired_touch_names[i])
        self.base_index = self.gym.find_actor_rigid_body_handle(
            self.env_handles[0], self.robot_handles[0], base_names[0])

    def reset(self, is_uniform_rollout=True):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # ================= for uniform rollouts ================= #
        if is_uniform_rollout:
            self.progress_buf[:] = torch.randint_like(self.progress_buf, low=0, high=self.max_episode_length)
        # ======================================================== #
        return super().reset()

    def reset_idx(self, env_ids):
        # convert env_id's dtype
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset dof position & velocity
        # ==================== randomize initial dof position ==================== #
        if self.is_randomized:
            positions_offset = torch_utils.torch_rand_float(
                self.rand_range_dof_pos[0], self.rand_range_dof_pos[1], (len(env_ids), self.num_dofs), device=self.device)
        else:
            positions_offset = torch.ones((len(env_ids), self.num_dofs), dtype=torch.float32, device=self.device)
        self.dof_positions[env_ids] = self.default_dof_positions[env_ids] * positions_offset
        self.dof_velocities[env_ids] = 0.0
        # ======================================================================== #
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_states),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # reset robot's base frame pose
        self.root_states[env_ids] = self.base_init_state
        # ==================== randomize initial root state ==================== #
        if self.is_randomized:
            initial_lin_vel = torch_utils.torch_rand_float(
                self.rand_range_root_vel[0], self.rand_range_root_vel[1], (len(env_ids), 3), device=self.device)
            initial_ang_vel = torch_utils.torch_rand_float(
                self.rand_range_root_vel[0], self.rand_range_root_vel[1], (len(env_ids), 3), device=self.device)
            self.root_states[env_ids, 7:10] += initial_lin_vel
            self.root_states[env_ids, 10:13] += initial_ang_vel
        # ====================================================================== #
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # ==== apply randomization ==== #
        if self.is_randomized:
            self.randomize(env_ids)
        # ============================= #

        # refresh tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset inner variables
        self.joint_targets[env_ids] = self.dof_positions[env_ids] + self.motor_offsets[env_ids]
        self.prev_joint_targets[env_ids] = self.joint_targets[env_ids].clone()
        self.prev_prev_joint_targets[env_ids] = self.joint_targets[env_ids].clone()
        self.prev_actions[env_ids] = (self.joint_targets[env_ids] - self.default_dof_positions[env_ids])/self.action_scale

        # reset buffers
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.stage_buf[env_ids] = 0.0
        self.stage_buf[env_ids, 0] = 1.0
        self.is_half_turn_buf[env_ids] = 0
        self.is_one_turn_buf[env_ids] = 0
        self.verified_turn_buf[env_ids] = 0
        self.start_time_buf[env_ids] = torch_utils.torch_rand_float(0.0, 5.0, (len(env_ids), 1), device=self.device).squeeze()
        self.cmd_time_buf[env_ids] = 0.0
        self.land_time_buf[env_ids] = 0.0
        self.takeoff_counter_buf[env_ids] = 0
        self.air_time_buf[env_ids] = 0.0
        self.landing_stable_counter_buf[env_ids] = 0
        self.landing_success_buf[env_ids] = 0
        self.flip_angle_buf[env_ids] = 0
        self.prev_pitch_buf[env_ids] = 0
        self.prev_vertical_velocity[env_ids] = 0
        self.last_contact_angle_buf[env_ids] = 0
        self.air_rotation_buf[env_ids] = 0
        self.true_air_rotation_buf[env_ids] = 0
        self.peak_true_air_rotation_buf[env_ids] = 0
        self.previous_free_air_buf[env_ids] = False
        self.early_tuck_weight_buf[env_ids] = 0
        self.jump_time_buf[env_ids] = 0
        for i in range(len(self.lag_joint_target_buffer)):
            self.lag_joint_target_buffer[i][env_ids, :] = self.joint_targets[env_ids]
        for i in range(len(self.lag_imu_buffer)):
            self.lag_imu_buffer[i][env_ids, :] = self.est_base_body_orns[env_ids]

        # estimate observations
        self.est_base_body_orns[env_ids] = torch_utils.quat_rotate_inverse(self.base_quaternions[env_ids], self.world_z[env_ids])
        self.est_dof_positions[env_ids] = self.dof_positions[env_ids] + self.motor_offsets[env_ids]
        self.est_dof_velocities[env_ids] = self.dof_velocities[env_ids]
        if self.is_randomized:
            self.est_base_body_orns[env_ids] += torch_utils.torch_rand_float(
                -self.noise_range_body_orn, self.noise_range_body_orn, (len(env_ids), 3), device=self.device)
            self.est_base_body_orns[env_ids] /= torch.norm(self.est_base_body_orns[env_ids], dim=-1, keepdim=True)
            self.est_dof_positions[env_ids] += torch_utils.torch_rand_float(
                -self.noise_range_dof_pos, self.noise_range_dof_pos, (len(env_ids), self.num_dofs), device=self.device)
            self.est_dof_velocities[env_ids] += torch_utils.torch_rand_float(
                -self.noise_range_dof_vel, self.noise_range_dof_vel, (len(env_ids), self.num_dofs), device=self.device)

        # calculate commands
        commands = torch.zeros((len(env_ids), 3), dtype=torch.float32, device=self.device)
        masks0 = (self.cmd_time_buf[env_ids] == 0).type(torch.float32)
        masks1 = (1.0 - masks0)*(self.progress_buf[env_ids]*self.control_dt < self.cmd_time_buf[env_ids] + 0.2).type(torch.float32)
        masks2 = (1.0 - masks0)*(1.0 - masks1)
        commands[:, 0] = masks0
        commands[:, 1] = masks1
        commands[:, 2] = masks2

        # reset observation
        obs = jit_compute_observations(
            self.est_base_body_orns[env_ids], self.est_dof_positions[env_ids], self.est_dof_velocities[env_ids],
            self.prev_actions[env_ids], commands)
        for history_idx in range(self.history_len):
            self.obs_buf[env_ids, history_idx*self.raw_obs_dim:(history_idx+1)*self.raw_obs_dim] = obs

        # reset state
        contact_forces = self.contact_forces[env_ids]
        foot_contact_forces = contact_forces[:, self.foot_indices, :]
        calf_contact_forces = contact_forces[:, self.calf_indices, :]
        self.states_buf[env_ids] = jit_compute_states(
            self.base_quaternions[env_ids], self.base_lin_vels[env_ids], self.base_ang_vels[env_ids], self.base_positions[env_ids],
            foot_contact_forces, calf_contact_forces, self.gravity,
            self.friction_coeffs[env_ids], self.restitution_coeffs[env_ids], self.stage_buf[env_ids])

        if self.viewer != None:
            cam_pos = torch.tensor(np.array([[0.0, 3.0, 1.0]]), dtype=torch.float32, device=self.device)
            cam_pos += self.base_positions
            cam_pos = gymapi.Vec3(*cam_pos[0])
            cam_target = gymapi.Vec3(*self.base_positions[0])
            self.gym.viewer_camera_look_at(
                self.viewer, self.env_handles[0], cam_pos, cam_target)

    def step(self, actions: torch.Tensor):
        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.pre_physics_step(action_tensor)
        if self.use_pose_guidance and 'stiffness' in self.pose_cfg:
            stiffness = (self.stage_buf @ self.stage_stiffness).unsqueeze(-1)
            damping = (self.stage_buf @ self.stage_damping).unsqueeze(-1)
            if self.use_early_tuck and self.early_tuck_cfg.get('blend_pd', True):
                weight = self.early_tuck_weight_buf.unsqueeze(-1)
                stiffness = (1-weight)*stiffness + weight*self.stage_stiffness[3]
                damping = (1-weight)*damping + weight*self.stage_damping[3]
        else:
            stiffness = self.cfg['env']['control']['stiffness']
            damping = self.cfg['env']['control']['damping']

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            if self.force_render:
                self.render()

            # calculate torques using PD control
            self.lag_joint_target_buffer = self.lag_joint_target_buffer[1:] + [self.joint_targets]
            joint_targets = self.lag_joint_target_buffer[0]
            current_dof_positions = self.dof_positions + self.motor_offsets
            current_dof_velocities = self.dof_velocities
            torques = stiffness*(joint_targets - current_dof_positions) - damping*current_dof_velocities
            torques = torch.clip(
                torques*self.motor_strengths, -self.dof_torques_upper_limits.unsqueeze(0),
                self.dof_torques_upper_limits.unsqueeze(0))
            if self.use_motor_model:
                torques = jit_motor_torque_envelope(torques, current_dof_velocities,
                                                   self.dof_torques_upper_limits, self.motor_no_load_speed)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()
        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def pre_physics_step(self, actions: torch.Tensor):
        self.prev_vertical_velocity[:] = self.base_lin_vels[:, 2]
        # update previous actions
        self.prev_actions[:] = actions
        self.prev_prev_joint_targets[:] = self.prev_joint_targets
        self.prev_joint_targets[:] = self.joint_targets

        # set PD targets
        smooth_weight = self.action_smooth_weight
        self.early_tuck_weight_buf.zero_()
        if self.use_early_tuck:
            self.early_tuck_weight_buf[:] = jit_early_tuck_weight(
                self.stage_buf[:, 2].bool(), self.takeoff_counter_buf,
                self.contact_forces, self.base_lin_vels[:, 2],
                int(self.early_tuck_cfg.get('ramp_steps', 2)),
                float(self.pose_cfg.get('takeoff_min_upward_velocity', 0.20)))
        policy_targets = actions*self.action_scale + self.default_dof_positions
        if self.use_pose_guidance:
            references = self._pose_reference()
            blends = (self.stage_buf @ self.pose_blends).unsqueeze(-1)
            if self.pose_cfg.get('landing_assistance', False):
                release = ((self.flip_angle_buf - self.pose_cfg['untuck_start_angle']) /
                           (self.pose_cfg['untuck_end_angle'] - self.pose_cfg['untuck_start_angle'])).clamp(0, 1)
                blends += (self.stage_buf[:, 3]*release*(self.pose_blends[4]-self.pose_blends[3])).unsqueeze(-1)
            if self.use_early_tuck:
                weight = self.early_tuck_weight_buf.unsqueeze(-1)
                references = (1-weight)*references + weight*self.stage_pose_targets[3]
                blends = (1-weight)*blends + weight*self.pose_blends[3]
            policy_targets = (1.0 - blends)*policy_targets + blends*references
        self.joint_targets[:] = smooth_weight*policy_targets \
                                + (1.0 - smooth_weight)*self.joint_targets
        if self.cfg['env']['control'].get('clip_position_targets', False):
            self.joint_targets[:] = torch.maximum(self.dof_pos_lower_limits,
                                                 torch.minimum(self.joint_targets, self.dof_pos_upper_limits))

    def _pose_reference(self):
        references = self.stage_buf @ self.stage_pose_targets
        if self.pose_cfg.get('rear_push_delay', 0) > 0:
            push = ((self.jump_time_buf-self.pose_cfg['rear_push_delay']) /
                    self.pose_cfg['rear_push_ramp']).clamp(0, 1).unsqueeze(-1)
            rear = self.rear_dof_indices
            rear_target = (1-push)*self.stage_pose_targets[1, rear] + push*self.stage_pose_targets[2, rear]
            references[:, rear] = torch.where(self.stage_buf[:, 2:3].bool(), rear_target, references[:, rear])
        # Begin opening before upright, giving the legs time to prepare for
        # contact. Progress comes from measured rotation, never elapsed time.
        release = ((self.flip_angle_buf - self.pose_cfg['untuck_start_angle']) /
                   (self.pose_cfg['untuck_end_angle'] - self.pose_cfg['untuck_start_angle'])).clamp(0, 1)
        release *= self.stage_buf[:, 3]
        return (1.0 - release.unsqueeze(-1))*references + release.unsqueeze(-1)*self.stage_pose_targets[4]

    def post_physics_step(self):
        """
        After refresh tensors (DoF, root, contact information), need to peform followings:
        1. check termination.
        2. make observation.
        3. calculate reward.
        4. calculate costs.
        5. call 'self.reset_idx' after max_episode_len.
        6. update inner variables, such as pre_actions.
        """
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.progress_buf += 1
        self.common_step_counter += 1

        # ==== apply randomization ==== #
        if self.is_randomized:
            self.randomize()
        # ============================= #

        # stage 0: stand, stage 1: down, stage 2: jump, stage 3: back turn, stage 4: land
        # =================== calculate rewards =================== #
        # com height
        com_height = self.base_positions[:, 2]
        self.rew_buf[:, 0] =  self.stage_buf[:, 0]*(-torch.abs(com_height - self.stand_height))
        self.rew_buf[:, 0] += self.stage_buf[:, 1]*(-torch.abs(com_height - self.sit_height))
        self.rew_buf[:, 0] += self.stage_buf[:, 2]*(com_height <= self.jump_reward_height_cap)*(com_height)
        self.rew_buf[:, 0] += self.stage_buf[:, 3]*(com_height <= self.jump_reward_height_cap)*(com_height)
        self.rew_buf[:, 0] += self.stage_buf[:, 4]*(-torch.abs(com_height - self.stand_height))
        # body balance
        body_z = torch_utils.quat_rotate_inverse(self.base_quaternions, self.world_z)
        rotating = (self.stage_buf[:, 2] + self.stage_buf[:, 3]) > 0
        pitch, progress = jit_backflip_progress(body_z, self.prev_pitch_buf, self.flip_angle_buf, rotating)
        angular_delta = progress - self.flip_angle_buf
        free_air = (self.stage_buf[:, 3] == 1) & (torch.norm(self.contact_forces, dim=-1).amax(dim=-1) <= 1.0)
        air_rotation, air_peak, new_air_progress = jit_true_air_progress(
            angular_delta, self.true_air_rotation_buf, self.peak_true_air_rotation_buf,
            self.previous_free_air_buf, free_air)
        self.true_air_rotation_buf[:] = air_rotation
        self.peak_true_air_rotation_buf[:] = air_peak
        self.previous_free_air_buf[:] = free_air
        self.flip_angle_buf[:] = progress
        self.verified_turn_buf[:] |= jit_backflip_complete(
            free_air, progress, air_rotation, self.minimum_air_rotation).long()
        self.prev_pitch_buf[:] = pitch
        self.rew_buf[:, 1] =  self.stage_buf[:, 0]*(-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        self.rew_buf[:, 1] += self.stage_buf[:, 1]*(-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        self.rew_buf[:, 1] += self.stage_buf[:, 2]*(-torch.abs(torch.arccos(torch.clamp(body_z[:, 1], -1.0, 1.0)) - np.pi/2.0))
        self.rew_buf[:, 1] += self.stage_buf[:, 3]*(-torch.abs(torch.arccos(torch.clamp(body_z[:, 1], -1.0, 1.0)) - np.pi/2.0))
        self.rew_buf[:, 1] += self.stage_buf[:, 4]*(-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        # pitch vel
        base_lin_vels = torch_utils.quat_rotate_inverse(self.base_quaternions, self.base_lin_vels)
        base_ang_vels = torch_utils.quat_rotate_inverse(self.base_quaternions, self.base_ang_vels)
        vel_penalty = torch.square(base_lin_vels[:, 0]) + torch.square(base_lin_vels[:, 1]) + torch.square(base_ang_vels[:, 2])
        base_ang_vel_y = base_ang_vels[:, 1]
        landing_vel_penalty = torch.square(base_lin_vels).sum(dim=-1) \
                              + torch.square(base_ang_vels).sum(dim=-1)
        self.rew_buf[:, 2] =  self.stage_buf[:, 0]*(-vel_penalty)
        self.rew_buf[:, 2] += self.stage_buf[:, 1]*(-vel_penalty)
        self.rew_buf[:, 2] += self.stage_buf[:, 2]*(1.0 - self.is_one_turn_buf)*(-base_ang_vel_y)
        self.rew_buf[:, 2] += self.stage_buf[:, 3]*(1.0 - self.is_one_turn_buf)*(-base_ang_vel_y)
        self.rew_buf[:, 2] += self.stage_buf[:, 4]*(-landing_vel_penalty)
        # energy
        self.rew_buf[:, 3] = -torch.square(self.dof_torques).mean(dim=-1)
        # style
        self.rew_buf[:, 4]  = -torch.square(self.dof_positions - self.default_dof_positions).mean(dim=-1)
        if self.use_pose_guidance:
            # Saturate smoothly instead of deleting the height reward above
            # the old 0.45 m ceiling. Jump needs both lift and angular impulse.
            flight = self.stage_buf[:, 2] + self.stage_buf[:, 3]
            height_reward = torch.clamp(com_height, max=self.jump_reward_height_cap)
            old_height_reward = (com_height <= self.jump_reward_height_cap)*com_height
            self.rew_buf[:, 0] += flight*(height_reward - old_height_reward)
            self.rew_buf[:, 0] += self.stage_buf[:, 2]*self.pose_cfg['upward_velocity_weight']*torch.clamp(
                self.base_lin_vels[:, 2], min=-1.0, max=self.pose_cfg['target_upward_velocity'])
            # Reward positive progress while airborne; a fast but tiny hop is
            # not sufficient. Turn/landing bonuses are added at their events.
            self.rew_buf[:, 2] += self.stage_buf[:, 3]*self.pose_cfg['rotation_progress_weight']*self.flip_angle_buf.clamp(0, 2*np.pi)
            self.rew_buf[:, 4] = -torch.square(self.dof_positions - self._pose_reference()).mean(dim=-1)
        if self.use_launch_training:
            # No recurring reward merely for holding the body high on the
            # ground. Signed lift rewards extension, not a stationary pose.
            launch_reward, launch_pitch_reward = jit_launch_rewards(self.base_lin_vels[:, 2], -base_ang_vel_y)
            self.rew_buf[:, 0] = torch.where(self.stage_buf[:, 2].bool(), launch_reward, self.rew_buf[:, 0])
            self.rew_buf[:, 2] = torch.where(self.stage_buf[:, 2].bool(),
                                            launch_pitch_reward, self.rew_buf[:, 2])
        # ========================================================= #
        # ==================== calculate costs ==================== #
        if self.use_flight_training:
            # Reward NEW signed rotation records during continuous verified
            # flight, not holding an angle, ground rolling, or oscillating.
            self.rew_buf[:, 2] = torch.where(self.stage_buf[:, 3].bool(),
                self.flight_cfg['progress_weight']*new_air_progress, self.rew_buf[:, 2])
            launch_rotation = self.flight_cfg['launch_rotation_weight'] * (-base_ang_vel_y).clamp(0, 8) * self.base_lin_vels[:, 2].clamp(0, 1)
            self.rew_buf[:, 2] += self.stage_buf[:, 2]*launch_rotation
        # foot contact
        foot_contact_threshold = 0.25
        foot_contact_forces = self.contact_forces[:, self.foot_indices, :]
        calf_contact_forces = self.contact_forces[:, self.calf_indices, :]
        foot_contact = ((torch.norm(foot_contact_forces, dim=2) > 10.0)|(torch.norm(calf_contact_forces, dim=2) > 10.0)).type(torch.float)
        self.air_rotation_buf[:] = self.flip_angle_buf - self.last_contact_angle_buf
        self.cost_buf[:, 0] =  self.stage_buf[:, 0]*(foot_contact_threshold)
        self.cost_buf[:, 0] += self.stage_buf[:, 1]*(foot_contact_threshold)
        self.cost_buf[:, 0] += self.stage_buf[:, 2]*(1.0 - (foot_contact[:, 2] + foot_contact[:, 3])/2.0)
        self.cost_buf[:, 0] += self.stage_buf[:, 3]*(foot_contact_threshold)
        self.cost_buf[:, 0] += self.stage_buf[:, 4]*(1.0 - foot_contact.mean(dim=-1))
        # body contact
        term_contact = torch.any(torch.norm(self.contact_forces[:, self.terminate_touch_indices, :], dim=-1) > 1.0, dim=-1)
        undesired_contact = torch.any(torch.norm(self.contact_forces[:, self.undesired_touch_indices, :], dim=-1) > 1.0, dim=-1)
        if self.use_preparation_training:
            real_feet = (torch.norm(foot_contact_forces, dim=-1) > 10).float()
            calf_touch = (torch.norm(calf_contact_forces, dim=-1) > 1).float()
            # Previous target is captured before this action, not after reset.
            target_delta = (self.joint_targets-self.prev_joint_targets)/self.action_scale
            prep_penalty, prep_support = jit_preparation_feedback(
                self.stage_buf, target_delta, self.base_lin_vels[:, 2], base_ang_vels,
                real_feet, calf_touch, term_contact, float(self.preparation_cfg['smooth_weight']),
                float(self.preparation_cfg['angular_weight']), float(self.preparation_cfg['vertical_weight']),
                float(self.preparation_cfg['support_weight']), float(self.preparation_cfg['calf_weight']))
            self.rew_buf[:, 2] -= prep_penalty
            self.rew_buf[:, 1] += prep_support
        if self.use_landing_training:
            real_feet = (torch.norm(foot_contact_forces, dim=-1) > 10).float()
            self.cost_buf[:, 0] = torch.where(self.stage_buf[:, 4] == 1,
                1-real_feet.mean(-1), self.cost_buf[:, 0])
            support_reward = jit_landing_support_reward(self.stage_buf[:, 4] == 1,
                self.flip_angle_buf, real_feet, body_z[:, 2],
                self.base_lin_vels, self.base_ang_vels, term_contact | undesired_contact)
            self.rew_buf[:, 1] += self.landing_cfg['support_weight']*support_reward
        self.cost_buf[:, 1] =  self.stage_buf[:, 0]*torch.logical_or(term_contact, undesired_contact).type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 1]*torch.logical_or(term_contact, undesired_contact).type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 2]*torch.logical_or(term_contact, undesired_contact).type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 3]*undesired_contact.type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 4]*undesired_contact.type(torch.float)
        # joint pos
        self.cost_buf[:, 2] = torch.mean((
            (self.dof_positions < self.dof_pos_lower_limits)|(self.dof_positions > self.dof_pos_upper_limits)
            ).to(torch.float), dim=-1)
        # joint vel
        self.cost_buf[:, 3] = torch.mean(
            (torch.abs(self.dof_velocities) > self.dof_vel_upper_limits).to(torch.float), dim=-1)
        # joint torque
        self.cost_buf[:, 4] = torch.mean(
            (torch.abs(self.dof_torques) > self.dof_torques_upper_limits).to(torch.float), dim=-1)
        # ========================================================= #

        # Update turn completion before evaluating the air-to-land transition.
        old_one_turn = self.is_one_turn_buf.clone()
        if self.strict_flight_validation:
            self.is_half_turn_buf[:] |= (self.flip_angle_buf >= np.pi).long()
            # Most of the rotation must occur after the last ground contact;
            # rolling halfway and hopping must not count as a backflip.
            self.is_one_turn_buf[:] |= jit_backflip_complete(
                self.stage_buf[:, 3].bool(), self.flip_angle_buf,
                self.air_rotation_buf, self.minimum_air_rotation).long()
            if self.use_pose_guidance:
                self.rew_buf[:, 2] += self.pose_cfg['full_turn_bonus']*(self.is_one_turn_buf - old_one_turn)
        elif not self.go1_transitions:
            self.is_half_turn_buf[:] = torch.logical_or(
                self.is_half_turn_buf.bool(), torch.logical_and(
                    body_z[:, 0] < 0, body_z[:, 2] < 0)).type(torch.long)
            self.is_one_turn_buf[:] = torch.logical_or(
                self.is_one_turn_buf.bool(), torch.logical_and(
                    self.is_half_turn_buf.bool(), torch.logical_and(
                        body_z[:, 0] >= 0, body_z[:, 2] >= 0))).type(torch.long)

        # Confirm takeoff over several consecutive control steps. A single
        # no-contact frame is common while the rear feet are still pushing.
        in_jump = self.stage_buf[:, 2] == 1.0
        self.jump_time_buf += in_jump.float()*self.control_dt
        feet_off = foot_contact.mean(dim=-1) < 0.1
        if self.strict_flight_validation:
            # Folded feet alone do not imply flight (the thighs may still
            # support the robot). Require no ground force on ANY body and
            # positive world-frame lift through the confirmation window.
            feet_off = jit_free_flight(self.contact_forces, self.base_lin_vels[:, 2],
                                      float(self.pose_cfg.get('takeoff_min_upward_velocity', 0.20)))
        self.takeoff_counter_buf[:] = torch.where(
            torch.logical_and(in_jump, feet_off),
            self.takeoff_counter_buf + 1,
            torch.zeros_like(self.takeoff_counter_buf),
        )

        # Track elapsed air-stage time, including the brief launch grace.
        in_air = self.stage_buf[:, 3] == 1.0
        self.air_time_buf[:] += in_air.type(torch.float32) * self.control_dt
        landing_contacts = (torch.norm(foot_contact_forces, dim=-1) > 10.0).float() if self.strict_flight_validation else foot_contact
        any_foot_contact = landing_contacts.mean(dim=-1) > 0.0
        self.last_contact_angle_buf[:] = torch.where(
            (torch.norm(self.contact_forces, dim=-1).amax(dim=-1) > 1.0) &
            ~self.is_one_turn_buf.bool(), self.flip_angle_buf, self.last_contact_angle_buf)
        descending = self.base_lin_vels[:, 2] < -self.landing_min_downward_velocity
        if self.strict_flight_validation:
            # The impact can reverse vertical velocity within this control
            # step. Include the pre-impact velocity when recognizing contact.
            descending |= self.prev_vertical_velocity < -self.landing_min_downward_velocity
        upright = body_z[:, 2] >= self.landing_upright_cos
        air_grace_complete = self.air_time_buf >= self.min_air_time

        # Update stages in reverse order so an environment advances at most one
        # stage per control step.
        from3_to4 = torch.logical_and(
            in_air, torch.logical_and(
                air_grace_complete, torch.logical_and(
                    any_foot_contact, torch.logical_and(
                        self.is_one_turn_buf.bool(), torch.logical_and(descending, upright)
                    )
                )
            )
        ).type(torch.float32)
        if self.go1_transitions:
            _, go1_land, _ = jit_go1_contact_gates(self.stage_buf, foot_contact, self.is_half_turn_buf)
            from3_to4 = go1_land.float()
        self.stage_buf[:, 3] = (1.0 - from3_to4)*self.stage_buf[:, 3]
        self.stage_buf[:, 4] = from3_to4 + (1.0 - from3_to4)*self.stage_buf[:, 4]
        from2_to3 = torch.logical_and(
            self.stage_buf[:, 2] == 1.0,
            self.takeoff_counter_buf >= self.takeoff_confirm_steps,
        ).type(torch.float32)
        if self.go1_transitions:
            go1_air, _, _ = jit_go1_contact_gates(self.stage_buf, foot_contact, self.is_half_turn_buf)
            from2_to3 = go1_air.float()
        self.stage_buf[:, 2] = (1.0 - from2_to3)*self.stage_buf[:, 2]
        self.stage_buf[:, 3] = from2_to3 + (1.0 - from2_to3)*self.stage_buf[:, 3]
        if self.use_launch_training:
            self.rew_buf[:, 0] += self.launch_cfg['takeoff_bonus']*from2_to3
        from1_to2 = torch.logical_and(
            self.stage_buf[:, 1] == 1.0, torch.logical_and(
                com_height <= self.sit_transition_height,
                foot_contact.mean(dim=-1) >= 0.9
            )
        ).type(torch.float32)
        self.stage_buf[:, 1] = (1.0 - from1_to2)*self.stage_buf[:, 1]
        self.stage_buf[:, 2] = from1_to2 + (1.0 - from1_to2)*self.stage_buf[:, 2]
        from0_to1 = torch.logical_and(
            self.stage_buf[:, 0] == 1.0, torch.logical_and(
                self.progress_buf*self.control_dt > self.start_time_buf, torch.logical_and(
                    com_height >= self.stand_transition_height,
                    self.is_half_turn_buf == 0
                )
            )
        ).type(torch.float32)
        self.stage_buf[:, 0] = (1.0 - from0_to1)*self.stage_buf[:, 0]
        self.stage_buf[:, 1] = from0_to1 + (1.0 - from0_to1)*self.stage_buf[:, 1]

        # Go1 updates quadrant flags AFTER stage transitions, before failures.
        if self.go1_transitions:
            self.is_half_turn_buf[:] |= ((body_z[:, 0] < 0) & (body_z[:, 2] < 0)).long()
            self.is_one_turn_buf[:] |= (self.is_half_turn_buf.bool() &
                                       (body_z[:, 0] >= 0) & (body_z[:, 2] >= 0)).long()

        # A landing is successful after all feet remain down while the body is
        # upright and slow for the configured duration. Success is reported but
        # does not reset the episode, making playback easier to inspect.
        stable_landing = torch.logical_and(
            self.stage_buf[:, 4] == 1.0, torch.logical_and(
                landing_contacts.mean(dim=-1) >= 0.9, torch.logical_and(
                    upright, torch.logical_and(
                        torch.abs(self.base_lin_vels[:, 2]) <= self.landing_max_vertical_speed,
                        torch.norm(self.base_ang_vels, dim=-1) <= self.landing_max_angular_speed,
                    )
                )
            )
        )
        if self.strict_flight_validation:
            stable_landing &= torch.norm(self.base_lin_vels[:, :2], dim=-1) <= self.landing_max_horizontal_speed
            stable_landing &= ~term_contact & ~undesired_contact
        if self.go1_transitions:
            # Independent evaluation: stage land / quadrant flags alone do not
            # establish a real airborne backflip or a stable feet-only landing.
            stable_landing &= self.verified_turn_buf.bool()
            stable_landing &= (torch.norm(foot_contact_forces, dim=-1) > 10).float().mean(-1) >= .9
            stable_landing &= torch.norm(self.base_lin_vels[:, :2], dim=-1) <= self.landing_max_horizontal_speed
            stable_landing &= ~term_contact & ~undesired_contact
        self.landing_stable_counter_buf[:] = torch.where(
            stable_landing,
            self.landing_stable_counter_buf + 1,
            torch.zeros_like(self.landing_stable_counter_buf),
        )
        old_landing_success = self.landing_success_buf.clone()
        self.landing_success_buf[:] = torch.logical_or(
            self.landing_success_buf.bool(),
            self.landing_stable_counter_buf >= self.landing_stable_steps,
        ).type(torch.long)
        if self.use_pose_guidance:
            self.rew_buf[:, 1] += self.pose_cfg['landing_bonus']*(self.landing_success_buf - old_landing_success)
        if self.use_landing_training:
            self.rew_buf[:, 1] += self.landing_cfg['success_bonus']*(self.landing_success_buf - old_landing_success)
        land_masks = torch.logical_and(self.land_time_buf == 0, self.stage_buf[:, 4] == 1).type(torch.float32)
        self.land_time_buf[:] = land_masks*(self.progress_buf*self.control_dt) + (1.0 - land_masks)*self.land_time_buf
        cmd_masks = torch.logical_and(self.cmd_time_buf == 0, self.stage_buf[:, 1] == 1).type(torch.float32)
        self.cmd_time_buf[:] = cmd_masks*(self.progress_buf*self.control_dt) + (1.0 - cmd_masks)*self.cmd_time_buf

        # check termination
        body_contacts = torch.any(torch.norm(self.contact_forces[:, self.terminate_touch_indices, :], dim=-1) > 1.0, dim=-1)
        landing_wo_turns = torch.logical_and(
            self.stage_buf[:, 3] == 1.0, torch.logical_and(
                air_grace_complete, torch.logical_and(
                    any_foot_contact, torch.logical_and(
                        descending, torch.logical_not(self.is_half_turn_buf.bool())
                    )
                )
            )
        )
        if self.strict_flight_validation:
            # After launch grace, touching down before completing the flip
            # is a failed attempt even when already past the half-turn mark.
            # Otherwise a robot can stay labelled air while rolling on ground.
            any_ground_contact = torch.norm(self.contact_forces, dim=-1).amax(dim=-1) > 1.0
            landing_wo_turns = ((self.stage_buf[:, 3] == 1.0) & air_grace_complete &
                               any_ground_contact & ~self.is_one_turn_buf.bool())
            body_contacts |= ((self.stage_buf[:, 3] + self.stage_buf[:, 4]) > 0) & undesired_contact
        self.fail_buf[:] = torch.logical_or(body_contacts, landing_wo_turns).type(torch.long)
        if self.go1_transitions:
            _, _, landing_wo_turns = jit_go1_contact_gates(self.stage_buf, foot_contact, self.is_half_turn_buf)
            self.fail_buf[:] = (body_contacts | landing_wo_turns).long()
        launch_timeout = torch.zeros_like(self.fail_buf, dtype=torch.bool)
        if self.use_launch_training:
            launch_timeout = jit_launch_timeout(
                self.stage_buf, self.jump_time_buf, self.progress_buf*self.control_dt,
                self.start_time_buf, float(self.launch_cfg['jump_timeout_s']),
                float(self.launch_cfg['command_timeout_s']))
            self.fail_buf[:] |= launch_timeout.long()
            self.rew_buf[:, 0] -= self.launch_cfg['timeout_penalty']*launch_timeout.float()

        # calculate reset buffer
        self.reset_buf[:] = torch.where(
            self.progress_buf >= self.max_episode_length,
            torch.ones_like(self.reset_buf), self.fail_buf
        )

        # estimate observations
        est_base_body_orns = torch_utils.quat_rotate_inverse(self.base_quaternions, self.world_z)
        self.est_dof_positions = self.dof_positions + self.motor_offsets
        self.est_dof_velocities = self.dof_velocities
        if self.is_randomized:
            est_base_body_orns += torch_utils.torch_rand_float(
                -self.noise_range_body_orn, self.noise_range_body_orn, (self.num_envs, 3), device=self.device)
            est_base_body_orns /= torch.norm(est_base_body_orns, dim=-1, keepdim=True)
            self.est_dof_positions += torch_utils.torch_rand_float(
                -self.noise_range_dof_pos, self.noise_range_dof_pos, (self.num_envs, self.num_dofs), device=self.device)
            self.est_dof_velocities += torch_utils.torch_rand_float(
                -self.noise_range_dof_vel, self.noise_range_dof_vel, (self.num_envs, self.num_dofs), device=self.device)
        self.lag_imu_buffer = self.lag_imu_buffer[1:] + [est_base_body_orns]
        self.est_base_body_orns[:] = self.lag_imu_buffer[0]

        # calculate commands
        commands = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        masks0 = (self.cmd_time_buf == 0).type(torch.float32)
        masks1 = (1.0 - masks0)*(self.progress_buf*self.control_dt < self.cmd_time_buf + 0.2).type(torch.float32)
        masks2 = (1.0 - masks0)*(1.0 - masks1)
        commands[:, 0] = masks0
        commands[:, 1] = masks1
        commands[:, 2] = masks2

        # update observation buffer
        obs = jit_compute_observations(
            self.est_base_body_orns, self.est_dof_positions, self.est_dof_velocities,
            self.prev_actions, commands)
        self.obs_buf[:, :-self.raw_obs_dim] = self.obs_buf[:, self.raw_obs_dim:].clone()
        self.obs_buf[:, -self.raw_obs_dim:] = obs

        # update state buffer
        self.states_buf[:] = jit_compute_states(
            self.base_quaternions, self.base_lin_vels, self.base_ang_vels, self.base_positions,
            foot_contact_forces, calf_contact_forces, self.gravity, self.friction_coeffs, self.restitution_coeffs, self.stage_buf)

        # return extra
        self.extras['costs'] = self.cost_buf.clone()
        self.extras['fails'] = self.fail_buf.clone()
        self.extras['fail_body_contact'] = body_contacts.clone()
        self.extras['fail_launch_timeout'] = launch_timeout.clone()
        self.extras['fail_landing_without_turn'] = landing_wo_turns.clone()
        self.extras['air_time'] = self.air_time_buf.clone()
        self.extras['early_tuck_weight'] = self.early_tuck_weight_buf.clone()
        self.extras['landing_success'] = self.landing_success_buf.clone()
        self.extras['verified_turn'] = self.verified_turn_buf.clone()
        # Capture diagnostics before reset_idx overwrites the root/DOF state.
        self.extras['base_vertical_velocity'] = self.base_lin_vels[:, 2].clone()
        self.extras['foot_contact'] = foot_contact.clone()
        self.extras['takeoff_counter'] = self.takeoff_counter_buf.clone()
        self.extras['base_quaternion'] = self.base_quaternions.clone()
        self.extras['joint_positions'] = self.dof_positions.clone()
        self.extras['joint_velocities'] = self.dof_velocities.clone()
        self.extras['joint_targets'] = self.joint_targets.clone()
        self.extras['half_turn'] = self.is_half_turn_buf.clone()
        self.extras['one_turn'] = self.is_one_turn_buf.clone()
        self.extras['flip_angle'] = self.flip_angle_buf.clone()
        self.extras['air_rotation'] = self.air_rotation_buf.clone()
        self.extras['peak_true_air_rotation'] = self.peak_true_air_rotation_buf.clone()
        self.extras['next_obs'] = self.obs_buf.clone()
        self.extras['next_states'] = self.states_buf.clone()
        self.extras['dones'] = self.reset_buf.clone()

        # reset
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0: self.reset_idx(env_ids)

    def randomize(self, env_ids=None):
        if self.common_step_counter % self.rand_period_gravity == 0:
            self.randomizeGravity()

        if env_ids is not None:
            self.randomizeMotorStrength(env_ids)
            self.randomizeMotorOffsets(env_ids)
        else:
            env_ids = (self.progress_buf % self.rand_period_motor_strength == 0).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.randomizeMotorStrength(env_ids)

    def randomizeMotorStrength(self, env_ids):
        self.motor_strengths[env_ids] = torch_utils.torch_rand_float(
            self.rand_range_motor_strength[0], self.rand_range_motor_strength[1], (len(env_ids), 1), device=self.device)

    def randomizeMotorOffsets(self, env_ids):
        self.motor_offsets[env_ids] = torch_utils.torch_rand_float(
            self.rand_range_motor_offset[0], self.rand_range_motor_offset[1], (len(env_ids), self.num_dofs), device=self.device)

    def randomizeGravity(self):
        sim_params = self.gym.get_sim_params(self.sim)
        gravity_noise = np.random.rand(3)*(self.rand_range_gravity[1] - self.rand_range_gravity[0]) + self.rand_range_gravity[0]
        gravity = np.array(self.cfg["sim"]["gravity"]) + gravity_noise
        self.gravity[:] = torch.tensor(gravity, dtype=torch.float32, device=self.device)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)

@torch.jit.script
def jit_preparation_feedback(stages, target_delta, vertical_vel, angular_vel,
                             real_feet, calf_touch, body_contact,
                             smooth_weight: float, angular_weight: float, vertical_weight: float,
                             support_weight: float, calf_weight: float):
    stand, sit = stages[:, 0], stages[:, 1]
    active = stand+sit
    penalty = active*smooth_weight*torch.square(target_delta).mean(-1)
    penalty += (stand+.5*sit)*angular_weight*torch.square(angular_vel[:, :2]).sum(-1)
    # Sit must be allowed to descend; only waiting penalizes vertical speed.
    penalty += stand*vertical_weight*torch.square(vertical_vel)
    support = active*(support_weight*real_feet.mean(-1)*(~body_contact).float()
                      - calf_weight*calf_touch.mean(-1))
    return penalty, support


@torch.jit.script
def jit_landing_support_reward(in_land, rotation, real_feet, upright, linear_vel, angular_vel, bad_contact):
    # Dense recovery feedback after substantial rotation, without prescribing
    # joint angles. Only independent verified success earns the event bonus.
    clean = in_land & (rotation >= 4.5) & ~bad_contact
    calm = torch.exp(-.5*torch.square(linear_vel).sum(-1)-.1*torch.square(angular_vel).sum(-1))
    return clean.float()*real_feet.mean(-1)*upright.clamp(0, 1)*calm


@torch.jit.script
def jit_go1_contact_gates(stages, foot_contact, half_turn):
    contact_fraction = foot_contact.mean(dim=-1)
    return ((stages[:, 2] == 1) & (contact_fraction < .1),
            (stages[:, 3] == 1) & (contact_fraction > 0) & (half_turn != 0),
            (stages[:, 3] == 1) & (contact_fraction > 0) & (half_turn == 0))


@torch.jit.script
def jit_early_tuck_weight(in_jump, counter, contact_forces, upward_velocity, ramp_steps: int, min_upward_velocity: float):
    free = (torch.norm(contact_forces, dim=-1).amax(dim=-1) < 1.0) & (upward_velocity > min_upward_velocity)
    return (in_jump & free).float() * (counter.float()/float(ramp_steps)).clamp(0, 1)


@torch.jit.script
def jit_true_air_progress(delta, rotation, peak, previous_free, current_free):
    continuous = previous_free & current_free
    rotation = torch.where(continuous, rotation + delta, torch.zeros_like(rotation))
    new_peak = torch.maximum(peak, rotation.clamp(0, 2*np.pi))
    return rotation, new_peak, new_peak - peak


@torch.jit.script
def jit_launch_rewards(vertical_velocity, backward_pitch_velocity):
    return 0.5*vertical_velocity.clamp(-2.0, 2.0) - 0.05, 0.1*backward_pitch_velocity.clamp(-3.0, 3.0)


@torch.jit.script
def jit_launch_timeout(stages, jump_time, elapsed, start_time, jump_limit: float, command_limit: float):
    # Evaluate AFTER transitions so a confirmed takeoff on the deadline wins.
    return ((stages[:, 2] == 1) & (jump_time >= jump_limit)) | (
        ((stages[:, 0] + stages[:, 1]) > 0) & (elapsed >= start_time + command_limit))


@torch.jit.script
def jit_motor_torque_envelope(request, velocity, peak_torque, no_load_speed: float):
    """Approximate motoring derating, with peak-current-limited braking.

    Not a calibrated electrical/thermal model. Do not clamp the simulated
    velocity: impacts/backdriving may exceed the motor's no-load speed.
    """
    upper = peak_torque*(1.0 - velocity/no_load_speed).clamp(0.0, 1.0)
    lower = -peak_torque*(1.0 + velocity/no_load_speed).clamp(0.0, 1.0)
    return torch.maximum(lower, torch.minimum(request, upper))


@torch.jit.script
def jit_backflip_complete(in_air, rotation, air_rotation, minimum_air_rotation: float):
    return in_air & (rotation >= 2*np.pi - 0.25) & (air_rotation >= minimum_air_rotation)


@torch.jit.script
def jit_free_flight(contact_forces, vertical_velocity, min_upward_velocity: float):
    return (torch.norm(contact_forces, dim=-1).amax(dim=-1) < 1.0) & (vertical_velocity > min_upward_velocity)


@torch.jit.script
def jit_backflip_progress(body_z, prev_pitch, progress, rotating):
    """Signed, unwrapped rotation around body Y; reject sideways singularities."""
    pitch = torch.atan2(body_z[:, 0], body_z[:, 2])
    delta = torch.atan2(torch.sin(pitch - prev_pitch), torch.cos(pitch - prev_pitch))
    valid = (torch.abs(body_z[:, 1]) < 0.5) & (torch.abs(delta) < 0.8)
    return pitch, progress + delta*(rotating & valid).float()


@torch.jit.script
def jit_compute_states(
    base_quaternions, base_lin_vels, base_ang_vels, base_positions,
    foot_contact_forces, calf_contact_forces, gravity, friction_coeffs, restitution_coeffs, stages,
):
    bb_lin_vels = torch_utils.quat_rotate_inverse(base_quaternions, base_lin_vels)
    bb_ang_vels = torch_utils.quat_rotate_inverse(base_quaternions, base_ang_vels)
    com_height = base_positions[:, 2:3]
    foot_contacts = ((torch.norm(foot_contact_forces, dim=2) > 1.0) \
                    | (torch.norm(calf_contact_forces, dim=2) > 1.0)).type(torch.float)
    gravities = gravity.unsqueeze(0).repeat(base_quaternions.shape[0], 1)
    states = torch.cat([
        bb_lin_vels, bb_ang_vels, com_height, foot_contacts,
        gravities, friction_coeffs, restitution_coeffs, stages], dim=-1)
    return states

@torch.jit.script
def jit_compute_observations(
    body_orns, dof_pos, dof_vel,
    prev_actions, commands,
):
    obs_list = []
    # body orientation
    obs_list.append(body_orns)
    # DoF's position and velocity
    obs_list.append(dof_pos)
    obs_list.append(dof_vel)
    # previous actions
    obs_list.append(prev_actions)
    # command
    obs_list.append(commands)
    # concatenate
    obs = torch.cat(obs_list, dim=-1)
    return obs
