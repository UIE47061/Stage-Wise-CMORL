"""Headless, deterministic checkpoint evaluation; does not modify model files.

Example:
    python scripts/diagnose_little_white.py --seed 2 --model_num 100000000
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Isaac Gym must be imported before torch.
from tasks import task_dict
from algos.comoppo.actor import Actor
from algos.comoppo.normalizer import ObsRMS
from utils import setSeed
from ruamel.yaml import YAML
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True, help='Checkpoint result seed.')
    parser.add_argument('--source_dir', type=Path, help='Optional explicit result directory for simulation-only transfer diagnostics.')
    parser.add_argument('--flight_source_dir', type=Path, help='Experimental second teacher for jump/air/land; simulation only.')
    parser.add_argument('--flight_model_num', type=int, default=100000000)
    parser.add_argument('--flight_stages', type=int, nargs='+', default=[2, 3, 4], choices=range(5),
                        help='Stages using the second teacher: 2=jump, 3=air, 4=land.')
    parser.add_argument('--default_pose_cfg', type=Path, help='Experimental action-origin alignment using another task default pose.')
    parser.add_argument('--model_num', type=int, required=True)
    parser.add_argument('--task_cfg_path', default='tasks/little_white_v3_backflip.yaml')
    parser.add_argument('--algo_cfg_path', default='algos/comoppo/little_white_v3_backflip.yaml')
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--steps', type=int, default=1500, help='Control steps per environment.')
    parser.add_argument('--eval_seed', type=int, default=42)
    parser.add_argument('--randomized', action='store_true', help='Use domain randomization from the task config.')
    parser.add_argument('--report', type=Path, help='Optional JSON output; must be a new file.')
    parser.add_argument('--trace', type=Path, help='JSON motion trace for the first environment; new file only.')
    parser.add_argument('--expect_takeoff', action='store_true', help='Exit nonzero if no confirmed takeoff.')
    parser.add_argument('--expect_success', action='store_true', help='Require at least 10 completed episodes and 80% successful landings.')
    args = parser.parse_args()
    if args.num_envs <= 0 or args.steps <= 0:
        parser.error('num_envs and steps must be positive')
    for path in (args.report, args.trace):
        if path and path.exists():
            parser.error(f'Output must be a new file: {path}')
    if args.report and args.trace and args.report.resolve() == args.trace.resolve():
        parser.error('report and trace must be different files')
    setSeed(args.eval_seed)
    with (ROOT / args.task_cfg_path).open() as f:
        cfg = YAML().load(f)
    backflip = cfg['env']['backflip']
    if args.expect_success and backflip.get('transition_mode') != 'go1' and not backflip.get('strict_flight_validation', backflip.get('pose_guidance', {}).get('enabled', False)):
        parser.error('--expect_success requires strict_flight_validation; legacy criteria are not backflip acceptance.')
    with (ROOT / args.algo_cfg_path).open() as f:
        algo = YAML().load(f)
    if args.default_pose_cfg:
        with (ROOT / args.default_pose_cfg).open() as f:
            source_pose = YAML().load(f)['env']['default_joint_positions']
        pose_map = {name.lower(): value for name, value in source_pose.items()}
        cfg['env']['default_joint_positions'] = {name: pose_map[name.lower()]
                                                for name in cfg['env']['default_joint_positions']}
    result_dir = (ROOT / args.source_dir if args.source_dir else
                  ROOT / 'results' / 'littlewhitev3backflip_comoppo' / f'seed_{args.seed}')
    checkpoint = result_dir / 'checkpoint' / f'model_{args.model_num}.pt'
    required = [checkpoint] + [result_dir / f'{name}_scale' / f'{args.model_num}.pkl' for name in ('obs', 'state')]
    if args.flight_source_dir:
        flight_source = ROOT / args.flight_source_dir
        required += [flight_source / 'checkpoint' / f'model_{args.flight_model_num}.pt']
        required += [flight_source / f'{name}_scale' / f'{args.flight_model_num}.pkl' for name in ('obs', 'state')]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg['env']['num_envs'] = args.num_envs
    cfg['env']['history_len'] = algo['history_len']
    cfg['env']['randomize']['is_randomized'] = args.randomized
    env = task_dict[cfg['name']](
        cfg=cfg, rl_device='cuda:0', sim_device='cuda:0', graphics_device_id=0,
        headless=True, virtual_screen_capture=False, force_render=False,
    )
    try:
        nstage = env.num_stages
        nstate = env.num_states - nstage
        actor = Actor(env.device, env.num_obs, nstate, nstage, env.num_acts,
                      -np.ones(env.num_acts), np.ones(env.num_acts), algo['model']['actor']).to(env.device)
        actor.load_state_dict(torch.load(str(checkpoint), map_location=env.device)['actor'])
        actor.eval()
        obs_rms = ObsRMS('obs', env.num_obs, env.history_len, env.device)
        state_rms = ObsRMS('state', nstate, 1, env.device)
        obs_rms.load(str(result_dir), args.model_num)
        state_rms.load(str(result_dir), args.model_num)
        flight_actor = None
        if args.flight_source_dir:
            flight_source = ROOT / args.flight_source_dir
            flight_actor = Actor(env.device, env.num_obs, nstate, nstage, env.num_acts,
                                 -np.ones(env.num_acts), np.ones(env.num_acts), algo['model']['actor']).to(env.device)
            flight_actor.load_state_dict(torch.load(str(flight_source/'checkpoint'/f'model_{args.flight_model_num}.pt'), map_location=env.device)['actor'])
            flight_actor.eval()
            flight_obs_rms = ObsRMS('obs', env.num_obs, env.history_len, env.device)
            flight_state_rms = ObsRMS('state', nstate, 1, env.device)
            flight_obs_rms.load(str(flight_source), args.flight_model_num)
            flight_state_rms.load(str(flight_source), args.flight_model_num)
        data = env.reset(is_uniform_rollout=False)
        counts = torch.zeros(nstage, device=env.device)
        heights = [[] for _ in range(nstage)]
        gate_samples = gate_height = gate_contacts = 0
        takeoffs = landings = episodes = failures = successes = 0
        jump_entries = 0
        body_failures = landing_failures = 0
        launch_timeouts = 0
        trace = []
        max_backward_rate = max_half_turns = max_full_turns = 0
        max_rotation = 0.0
        completed_air_rotations = []
        max_true_air_rotation = 0.0
        peak_joint_speed = 0.0
        peak_nonfailure_joint_speed = 0.0
        velocity_violations = position_violations = 0.0
        max_height = max_vz = max_air = 0.0
        with torch.no_grad():
            for step in range(args.steps):
                state = data['states']
                old_stage = state[:, -nstage:].argmax(dim=-1)
                action = actor(obs_rms.normalize(data['obs']), state_rms.normalize(state[:, :-nstage]),
                               state[:, -nstage:])[0]
                if flight_actor is not None:
                    flight_action = flight_actor(flight_obs_rms.normalize(data['obs']),
                                                 flight_state_rms.normalize(state[:, :-nstage]), state[:, -nstage:])[0]
                    use_second = torch.zeros_like(old_stage, dtype=torch.bool)
                    for selected_stage in args.flight_stages:
                        use_second |= old_stage == selected_stage
                    action = torch.where(use_second.unsqueeze(-1), flight_action, action)
                data, reward, _, info = env.step(action)
                end_state = info['next_states']
                if not all(bool(torch.isfinite(t).all()) for t in (end_state, data['obs'], action, reward,
                           info['costs'], info['joint_positions'], info['joint_velocities'], info['joint_targets'])):
                    raise RuntimeError('Non-finite rollout tensor')
                stage = end_state[:, -nstage:].argmax(dim=-1)
                h = end_state[:, 6]
                counts += end_state[:, -nstage:].sum(dim=0)
                for i in range(nstage):
                    heights[i].extend(h[stage == i].cpu().tolist())
                # Gate inputs belong to the stage BEFORE transition. Using
                # the resulting stage would omit every successful sit exit.
                sit = old_stage == 1
                gate_samples += int(sit.sum())
                gate_height += int((sit & (h <= env.sit_transition_height)).sum())
                gate_contacts += int((sit & (info['foot_contact'].mean(dim=-1) >= .9)).sum())
                takeoffs += int(((old_stage == 2) & (stage == 3)).sum())
                jump_entries += int(((old_stage == 1) & (stage == 2)).sum())
                landings += int(((old_stage == 3) & (stage == 4)).sum())
                episodes += int(info['dones'].sum())
                failures += int(info['fails'].sum())
                body_failures += int(info['fail_body_contact'].sum())
                launch_timeouts += int(info['fail_launch_timeout'].sum())
                landing_failures += int(info['fail_landing_without_turn'].sum())
                successes += int((info['landing_success'].bool() & info['dones'].bool()
                                  & ~info['fails'].bool()).sum())
                max_height = max(max_height, float(h.max()))
                max_vz = max(max_vz, float(info['base_vertical_velocity'].max()))
                max_air = max(max_air, float(info['air_time'].max()))
                max_backward_rate = max(max_backward_rate, float((-end_state[:, 4]).max()))
                max_half_turns += int((info['half_turn'].bool() & info['dones'].bool()).sum())
                max_full_turns += int((info['one_turn'].bool() & info['dones'].bool()).sum())
                max_rotation = max(max_rotation, float(info['flip_angle'].max()))
                max_true_air_rotation = max(max_true_air_rotation, float(info['peak_true_air_rotation'].max()))
                completed_air_rotations.extend(info['peak_true_air_rotation'][info['dones'].bool()].cpu().tolist())
                peak_joint_speed = max(peak_joint_speed, float(info['joint_velocities'].abs().max()))
                valid_speeds = info['joint_velocities'][~info['fails'].bool()]
                if valid_speeds.numel():
                    peak_nonfailure_joint_speed = max(peak_nonfailure_joint_speed, float(valid_speeds.abs().max()))
                velocity_violations += float(info['costs'][:, 3].sum())
                position_violations += float(info['costs'][:, 2].sum())
                if args.trace:
                    trace.append(dict(step=step, stage=int(stage[0]), state=end_state[0].cpu().tolist(),
                                      quaternion=info['base_quaternion'][0].cpu().tolist(),
                                      q=info['joint_positions'][0].cpu().tolist(),
                                      dq=info['joint_velocities'][0].cpu().tolist(),
                                      targets=info['joint_targets'][0].cpu().tolist(),
                                      contact=info['foot_contact'][0].cpu().tolist(),
                                      air_rotation=float(info['air_rotation'][0]),
                                      body_contact=int(info['fail_body_contact'][0]),
                                      flip_angle=float(info['flip_angle'][0]),
                                      fail=int(info['fails'][0]), done=int(info['dones'][0])))
        report = dict(
            randomized=env.is_randomized,
            transition_mode=backflip.get('transition_mode', 'legacy'),
            takeoff_count_definition=('Go1 air-stage entries; not proof of whole-body flight'
                                      if env.go1_transitions else 'configured takeoff confirmation'),
            checkpoint=str(checkpoint), task_config=args.task_cfg_path,
            flight_source_dir=str(args.flight_source_dir) if args.flight_source_dir else None,
            flight_model_num=args.flight_model_num if args.flight_source_dir else None,
            flight_stages=args.flight_stages if args.flight_source_dir else None,
            default_joint_positions=dict(cfg['env']['default_joint_positions']),
            algo_config_path=args.algo_cfg_path,
            backflip_config=dict(cfg['env']['backflip']),
            control_config=dict(cfg['env']['control']),
            eval_seed=args.eval_seed, num_envs=args.num_envs, control_steps=args.steps,
            stage_fraction=dict(zip(env.stage_names, (counts / counts.sum()).tolist())),
            stage_height_quantiles={name: np.quantile(values, [0, .5, 1]).tolist() if values else []
                                    for name, values in zip(env.stage_names, heights)},
            sit_gate=dict(height_threshold=env.sit_transition_height,
                          height_pass_fraction=gate_height / max(1, gate_samples),
                          contact_pass_fraction=gate_contacts / max(1, gate_samples)),
            confirmed_takeoffs=takeoffs, land_entries=landings, completed_episodes=episodes,
            jump_entries=jump_entries, takeoff_rate_per_jump_entry=takeoffs/max(1, jump_entries),
            failed_episodes=failures, successful_completed_episodes=successes,
            failure_causes=dict(body_contact=body_failures, landing_without_turn=landing_failures, launch_timeout=launch_timeouts),
            peak_base_height_m=max_height, peak_upward_velocity_m_s=max_vz,
            max_air_stage_time_s=max_air,
            peak_backward_pitch_rate_rad_s=max_backward_rate,
            episodes_with_half_turn=max_half_turns, episodes_with_full_turn=max_full_turns,
            peak_backward_rotation_deg=float(np.degrees(max_rotation)),
            flight_metric_version=1,
            mean_episode_true_air_rotation_deg=float(np.degrees(np.mean(completed_air_rotations))) if completed_air_rotations else 0.0,
            peak_true_air_rotation_deg=float(np.degrees(max_true_air_rotation)),
            true_air_half_turn_rate=float(np.mean(np.asarray(completed_air_rotations) >= np.pi)) if completed_air_rotations else 0.0,
            peak_joint_speed_rad_s=peak_joint_speed,
            peak_nonfailure_joint_speed_rad_s=peak_nonfailure_joint_speed,
            joint_velocity_violation_fraction=velocity_violations / (args.steps*args.num_envs),
            joint_position_violation_fraction=position_violations / (args.steps*args.num_envs),
            successful_landing_rate=successes / max(1, episodes),
        )
        print(json.dumps(report, indent=2), flush=True)
        if args.report:
            # Refuse to replace an earlier measurement.
            with args.report.open('x') as f:
                json.dump(report, f, indent=2)
        if args.trace:
            with args.trace.open('x') as f:
                json.dump(dict(joint_names=env.dof_names, control_dt=env.control_dt, frames=trace), f)
        if args.expect_takeoff and takeoffs == 0:
            raise SystemExit('No confirmed takeoff; do not treat this checkpoint as a working jump policy.')
        if args.expect_success and (episodes < 10 or successes / episodes < 0.8):
            raise SystemExit('Backflip acceptance FAILED: need >=10 completed episodes and >=80% successful landings.')
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
