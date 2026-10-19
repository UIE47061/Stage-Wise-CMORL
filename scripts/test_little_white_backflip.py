"""Regression checks for the active free-policy baseline (Isaac Gym venv)."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tasks.little_white_v3_backflip import Env, jit_free_flight, jit_backflip_complete, jit_go1_contact_gates, jit_landing_support_reward, jit_preparation_feedback
from algos.comoppo.actor import Actor
from ruamel.yaml import YAML
import numpy as np
import torch


class FreePolicyTests(unittest.TestCase):
    def test_preparation_feedback_is_local_and_allows_sitting_down(self):
        stages = torch.eye(5)
        delta = torch.ones(5,12)
        linear_z = torch.full((5,), -1.)
        omega = torch.ones(5,3)
        feet = torch.ones(5,4); calf = torch.zeros(5,4); body = torch.zeros(5,dtype=torch.bool)
        penalty, support = jit_preparation_feedback(stages,delta,linear_z,omega,feet,calf,body,.2,.1,1.,.1,.1)
        torch.testing.assert_close(penalty,torch.tensor([1.4,.3,0.,0.,0.]))
        torch.testing.assert_close(support,torch.tensor([.1,.1,0.,0.,0.]))
        no_vertical, _ = jit_preparation_feedback(stages,delta,torch.zeros(5),omega,feet,calf,body,.2,.1,1.,.1,.1)
        self.assertEqual(float(no_vertical[1]),float(penalty[1]))
        _, kneeling = jit_preparation_feedback(stages,delta,linear_z,omega,torch.zeros(5,4),torch.ones(5,4),body,.2,.1,1.,.1,.1)
        self.assertLess(float(kneeling[1]),float(support[1]))

    def test_landing_support_requires_rotation_clean_feet_and_calm_body(self):
        land = torch.ones(6, dtype=torch.bool)
        rotation = torch.tensor([5.,5.,5.,5.,0.,5.])
        feet = torch.ones(6,4); feet[1] = 0
        upright = torch.ones(6)
        linear = torch.zeros(6,3); angular = torch.zeros(6,3); angular[3,1] = 10
        bad = torch.zeros(6,dtype=torch.bool); bad[2] = True; land[5] = False
        r = jit_landing_support_reward(land,rotation,feet,upright,linear,angular,bad)
        self.assertEqual(float(r[0]),1.)
        self.assertLess(float(r[3]),.001)
        self.assertEqual(r[[1,2,4,5]].tolist(),[0.,0.,0.,0.])

    def test_go1_contact_gates_and_independent_turn_validation(self):
        stages = torch.eye(5)[torch.tensor([2, 2, 3, 3, 3, 4])]
        feet = torch.tensor([[0.,0,0,0], [0,0,1,1], [1,0,0,0],
                             [1,0,0,0], [0,0,0,0], [1,1,1,1]])
        half = torch.tensor([0,0,1,0,1,1])
        air, land, fail = jit_go1_contact_gates(stages, feet, half)
        self.assertEqual(air.tolist(), [True,False,False,False,False,False])
        self.assertEqual(land.tolist(), [False,False,True,False,False,False])
        self.assertEqual(fail.tolist(), [False,False,False,True,False,False])
        # Go1 land can be entered after a partial turn; strict evaluation rejects it.
        self.assertTrue(land[2])
        self.assertFalse(jit_backflip_complete(torch.tensor([True]),
            torch.tensor([3.2]), torch.tensor([2.0]), 4.5)[0])

    def test_action_mapping_is_policy_only_in_every_stage(self):
        task = YAML().load((ROOT/'tasks/little_white_v3_backflip.yaml').read_text())
        cfg = task['env']
        self.assertEqual(cfg['backflip']['transition_mode'], 'go1')
        self.assertFalse(cfg['backflip']['strict_flight_validation'])
        self.assertNotIn('pose_guidance', cfg['backflip'])
        self.assertNotIn('early_tuck', cfg['backflip'])
        default = torch.tensor(list(cfg['default_joint_positions'].values())).repeat(5, 1)
        before = default + .1
        action = torch.linspace(-2, 2, 60).reshape(5, 12)
        fake = SimpleNamespace(cfg=task, base_lin_vels=torch.zeros(5, 3),
            prev_vertical_velocity=torch.zeros(5), prev_actions=torch.zeros_like(action),
            prev_prev_joint_targets=before.clone(), prev_joint_targets=before.clone(),
            joint_targets=before.clone(), action_smooth_weight=cfg['control']['action_smooth_weight'],
            action_scale=cfg['control']['action_scale'], default_dof_positions=default,
            early_tuck_weight_buf=torch.zeros(5), use_early_tuck=False,
            use_pose_guidance=False, stage_buf=torch.eye(5))
        Env.pre_physics_step(fake, action)
        expected = .5*(default+.25*action)+.5*before
        torch.testing.assert_close(fake.joint_targets, expected)

    def test_all_policy_stages_can_change(self):
        cfg = YAML().load((ROOT/'algos/comoppo/little_white_v3_backflip.yaml').read_text())
        actor = Actor('cpu', 4, 2, 5, 12, -np.ones(12), np.ones(12), cfg['model']['actor'])
        self.assertTrue(all(p.requires_grad for p in actor.parameters()))
        self.assertFalse(any(k.startswith(('ground_', 'jump_reference_')) for k in actor.state_dict()))
        obs, state, phase = torch.randn(5, 4), torch.randn(5, 2), torch.eye(5)
        before = actor(obs, state, phase)[0].detach().clone()
        opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
        opt.zero_grad()
        actor(obs, state, phase)[0].sum().backward()
        opt.step()
        after = actor(obs, state, phase)[0].detach()
        self.assertTrue(((after-before).abs().sum(-1) > 0).all())

    def test_ground_contact_and_partial_turn_are_not_success(self):
        contact = torch.zeros(2, 6, 3)
        contact[0, 0, 2] = 10
        self.assertEqual(jit_free_flight(contact, torch.ones(2), .2).tolist(), [False, True])
        complete = jit_backflip_complete(torch.tensor([True, True, False]),
            torch.tensor([6.2, 6.2, 6.2]), torch.tensor([1., 4.6, 4.6]), 4.5)
        self.assertEqual(complete.tolist(), [False, True, False])


if __name__ == '__main__':
    unittest.main()
