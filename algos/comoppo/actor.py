from algos.common.network_base import MLP, initWeights
from algos.common.actor_base import (
    ActorBase, normalize, unnormalize, clip
)

from typing import Tuple
from copy import deepcopy
import numpy as np
import torch


class Actor(ActorBase):
    def __init__(self, device:torch.device, obs_dim:int, state_dim:int, phase_dim:int, action_dim:int, \
                action_bound_min:np.ndarray, action_bound_max:np.ndarray, actor_cfg:dict, \
                log_std_min:float=-4.0, log_std_max:float=2.0) -> None:
        ActorBase.__init__(self, device)

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.phase_dim = phase_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # for model
        self.actor_cfg = actor_cfg
        self.use_action_bound = self.actor_cfg['use_action_bound']
        if 'last_activation' in self.actor_cfg:
            self.last_activation = eval(f'torch.nn.{self.actor_cfg["last_activation"]}')()
        else:
            self.last_activation = lambda x: x
        self.activation = eval(f'torch.nn.{self.actor_cfg["mlp"]["activation"]}')
        self.log_std_init = self.actor_cfg['log_std_init']
        self.log_std_fix = self.actor_cfg['log_std_fix']

        # for action
        if self.use_action_bound:
            self.action_bound_min = torch.tensor(
                action_bound_min, device=device, dtype=torch.float32)
            self.action_bound_max = torch.tensor(
                action_bound_max, device=device, dtype=torch.float32)
        else:
            self.action_bound_min = torch.tensor(
                -np.ones(self.action_dim), device=device, dtype=torch.float32)
            self.action_bound_max = torch.tensor(
                np.ones(self.action_dim), device=device, dtype=torch.float32)

        # build model
        self.build()
        self.freeze_ground = bool(actor_cfg.get('freeze_ground', False))
        self.jump_residual_limit = float(actor_cfg.get('jump_residual_limit', 0.0))
        if not np.isfinite(self.jump_residual_limit) or self.jump_residual_limit < 0:
            raise ValueError('jump_residual_limit must be finite and nonnegative.')
        if self.jump_residual_limit and not self.freeze_ground:
            raise ValueError('Bounded jump adjustment requires freeze_ground.')
        if self.freeze_ground:
            if not self.log_std_fix:
                raise ValueError('freeze_ground requires fixed action standard deviation.')
            self.ground_model = deepcopy(self.model).requires_grad_(False)
            self.ground_mean_decoder = deepcopy(self.mean_decoder).requires_grad_(False)
        if self.jump_residual_limit:
            self.jump_reference_model = deepcopy(self.model).requires_grad_(False)
            self.jump_reference_mean_decoder = deepcopy(self.mean_decoder).requires_grad_(False)

    def load_state_dict(self, state_dict, strict=True):
        legacy = self.freeze_ground and not any(k.startswith('ground_') for k in state_dict)
        new_jump = self.jump_residual_limit > 0 and not any(k.startswith('jump_reference_') for k in state_dict)
        result = super().load_state_dict(state_dict, strict=False if legacy or new_jump else strict)
        if legacy or new_jump:
            allowed = (('ground_',) if legacy else ()) + (('jump_reference_',) if new_jump else ())
            unexpected_missing = [k for k in result.missing_keys if not k.startswith(allowed)]
            if unexpected_missing or result.unexpected_keys:
                raise RuntimeError('Incompatible actor checkpoint: '+str(result))
        if legacy:
            self.ground_model.load_state_dict(self.model.state_dict())
            self.ground_mean_decoder.load_state_dict(self.mean_decoder.state_dict())
        if new_jump:
            self.jump_reference_model.load_state_dict(self.model.state_dict())
            self.jump_reference_mean_decoder.load_state_dict(self.mean_decoder.state_dict())
        return result


    def build(self) -> None:
        self.add_module('model', MLP(
            input_size=(self.obs_dim + self.state_dim + self.phase_dim), 
            output_size=self.actor_cfg['mlp']['shape'][-1],
            shape=self.actor_cfg['mlp']['shape'][:-1], 
            activation=self.activation,
        ))
        self.add_module("mean_decoder", torch.nn.Sequential(
            self.activation(),
            torch.nn.Linear(self.actor_cfg['mlp']['shape'][-1], self.action_dim),
        ))
        if self.log_std_fix:
            self.std_decoder = lambda x: torch.ones(
                *x.shape[:-1], self.action_dim, dtype=torch.float, device=self.device)*self.log_std_init
        else:
            self.add_module("std_decoder", torch.nn.Sequential(
                self.activation(),
                torch.nn.Linear(self.actor_cfg['mlp']['shape'][-1], self.action_dim),
            ))
        
    def forward(self, obs:torch.Tensor, state:torch.Tensor, phase:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        output: (mean, log_std, std)
        '''
        x = self.model(torch.cat([obs, state, phase], dim=-1))
        mean = self.last_activation(self.mean_decoder(x))
        if self.freeze_ground:
            with torch.no_grad():
                ground_mean = self.last_activation(self.ground_mean_decoder(
                    self.ground_model(torch.cat([obs, state, phase], dim=-1))))
            if self.jump_residual_limit:
                with torch.no_grad():
                    reference_mean = self.last_activation(self.jump_reference_mean_decoder(
                        self.jump_reference_model(torch.cat([obs, state, phase], dim=-1))))
                jump_mean = ground_mean + self.jump_residual_limit*torch.tanh((mean-reference_mean)/self.jump_residual_limit)
                mean = torch.where((phase[..., 2] > .5).unsqueeze(-1), jump_mean, mean)
                frozen_stage_count = 2
            else:
                frozen_stage_count = 3
            mean = torch.where((phase[..., :frozen_stage_count].sum(-1) > .5).unsqueeze(-1), ground_mean, mean)
        log_std = self.std_decoder(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, log_std, std

    def updateActionDist(self, obs:torch.Tensor, state:torch.Tensor, phase:torch.Tensor, epsilon:torch.Tensor) -> None:
        self.action_mean, self.action_log_std, self.action_std = \
            self.forward(obs, state, phase)
        self.normal_action = self.action_mean + epsilon*self.action_std
        self.action_dist = torch.distributions.Normal(self.action_mean, self.action_std)

    def sample(self, deterministic:bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if deterministic:
            norm_action = self.action_mean
        else:
            norm_action = self.normal_action
        unnorm_action = unnormalize(norm_action, self.action_bound_min, self.action_bound_max)
        return norm_action, unnorm_action

    def getMeanStd(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.action_mean, self.action_std
    
    def getDist(self) -> torch.distributions.Distribution:
        return self.action_dist
        
    def getEntropy(self) -> torch.Tensor:
        '''
        return entropy of action distribution given state.
        '''
        entropy = torch.mean(torch.sum(self.action_dist.entropy(), dim=-1)) + 0.0*self.action_mean.mean()
        return entropy
    
    def getLogProb(self, action=None) -> torch.Tensor:
        '''
        return log probability of action given state.
        '''
        if action is None:
            log_prob = torch.sum(self.action_dist.log_prob(self.normal_action), dim=-1)
        else:
            log_prob = torch.sum(self.action_dist.log_prob(action), dim=-1)
        return log_prob

    def initialize(self) -> None:
        for name, module in self.named_children():
            if name.startswith(('ground_', 'jump_reference_')):
                continue
            if name == 'std_decoder':
                initializer = lambda m: initWeights(m, init_bias=self.log_std_init)
            else:
                initializer = lambda m: initWeights(m)
            module.apply(initializer)
        if self.freeze_ground:
            self.ground_model.load_state_dict(self.model.state_dict())
            self.ground_mean_decoder.load_state_dict(self.mean_decoder.state_dict())
        if self.jump_residual_limit:
            self.jump_reference_model.load_state_dict(self.model.state_dict())
            self.jump_reference_mean_decoder.load_state_dict(self.mean_decoder.state_dict())
