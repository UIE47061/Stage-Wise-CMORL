# env
from tasks import task_dict

# algorithm
from algos import algo_dict

# utils
from utils import backupFiles, setSeed, cprint
from utils.wrapper import EnvWrapper
from utils.slackbot import Slackbot
from utils.logger import Logger

# base
from ruamel.yaml import YAML
from copy import deepcopy
import numpy as np
import argparse
import torch
import wandb
import time
from pathlib import Path

def getParser():
    parser = argparse.ArgumentParser()
    # common
    parser.add_argument('--wandb', action='store_true', help='use wandb?')
    parser.add_argument('--slack', action='store_true', help='use slack?')
    parser.add_argument('--test', action='store_true', help='test or train?')
    parser.add_argument('--device_type', type=str, default='gpu', help='gpu or cpu.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index.')
    parser.add_argument('--model_num', type=int, default=0, help='num model.')
    parser.add_argument('--init_from', type=str, default=None,
                        help='Warm-start from this result directory and save to a NEW seed; step count restarts at zero.')
    parser.add_argument('--total_steps', type=int, default=None, help='Override the training step budget in task YAML.')
    parser.add_argument('--reset_reward_stats', action='store_true', help='Warm-start only: relearn reward normalization after changing rewards.')
    parser.add_argument('--reset_critics', action='store_true', help='Warm-start only: initialize value networks for a changed reward/task.')
    parser.add_argument('--reset_optimizers', action='store_true', help='Warm-start only: discard old optimizer moments and restore configured learning rates.')
    parser.add_argument('--console_freq', type=int, default=100000, help='Training steps between local progress reports; 0 disables.')
    parser.add_argument('--save_freq', type=int, default=int(1e7), help='# of time steps for save.')
    parser.add_argument('--wandb_freq', type=int, default=int(5e4), help='# of time steps for wandb logging.')
    parser.add_argument('--slack_freq', type=int, default=int(2.5e6), help='# of time steps for slack message.')
    parser.add_argument('--seed', type=int, default=1, help='seed number.')
    parser.add_argument('--task_cfg_path', type=str, help='cfg.yaml file location for task.')
    parser.add_argument('--algo_cfg_path', type=str, help='cfg.yaml file location for algorithm.')
    parser.add_argument('--project_name', type=str, default="Stage-Wise-CMORL", help='wandb project name.')
    parser.add_argument('--render',  action='store_true', help='rendering?')
    parser.add_argument('--comment', type=str, default=None, help='wandb comment saved in run name.')
    return parser

def require_checkpoint(save_dir, model_num):
    """Never silently play or resume randomly initialized weights."""
    root = Path(save_dir)
    required = [root / 'checkpoint' / f'model_{model_num}.pt']
    required += [root / f'{name}_scale' / f'{model_num}.pkl' for name in ('obs', 'state', 'reward')]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing checkpoint/normalization files: ' + ', '.join(missing))

def train(args, task_cfg, algo_cfg):
    if algo_cfg.get('model', {}).get('actor', {}).get('freeze_ground', False) and not (args.init_from or args.model_num):
        raise ValueError('freeze_ground requires --init_from or --model_num; do not freeze a randomly initialized ground policy.')
    if (args.reset_critics or args.reset_optimizers) and not args.init_from:
        raise ValueError('Critic/optimizer resets require --init_from; ordinary resume must preserve training state.')
    if args.reset_reward_stats and not args.init_from:
        raise ValueError('--reset_reward_stats requires --init_from; do not reset statistics on ordinary resume.')
    if args.total_steps is not None:
        task_cfg['n_total_steps'] = args.total_steps
    if task_cfg['n_total_steps'] <= 0 or task_cfg['n_total_steps'] % algo_cfg['n_steps']:
        raise ValueError('total_steps must be positive and divisible by algo n_steps.')
    if task_cfg['env']['num_envs'] <= 0 or algo_cfg['n_steps'] % task_cfg['env']['num_envs']:
        raise ValueError('algo n_steps must be divisible by num_envs; e.g. 10000 / 500.')
    if args.init_from:
        require_checkpoint(args.init_from, args.model_num)
        if Path(args.save_dir).exists():
            raise ValueError('Warm-start requires an unused destination seed to preserve existing results.')
    elif args.model_num:
        require_checkpoint(args.save_dir, args.model_num)
        if args.model_num >= task_cfg['n_total_steps']:
            raise ValueError('To resume, total_steps must exceed model_num; it is the cumulative target, not extra steps.')
    elif any((Path(args.save_dir) / 'checkpoint').glob('model_*.pt')):
        raise ValueError('This seed already has checkpoints. Use --model_num to resume or choose an unused seed.')
    # set seed
    setSeed(args.seed)
    task_cfg['env']['history_len'] = algo_cfg['history_len']

    # backup configurations
    backupFiles(f"{args.save_dir}/backup", task_cfg['backup_files'], algo_cfg['backup_files'])
    # Record overrides as well as source files, so short runs are reproducible.
    with open(f'{args.save_dir}/backup/effective_task.yaml', 'w') as f:
        YAML().dump(task_cfg, f)
    with open(f'{args.save_dir}/backup/run.yaml', 'w') as f:
        YAML().dump(vars(args), f)

    # create environments
    env_fn = lambda: task_dict[task_cfg['name']](
        cfg=task_cfg, rl_device=args.device_name, sim_device=args.device_name, 
        graphics_device_id=0, headless=(not args.render), 
        virtual_screen_capture=False, force_render=args.render
    )
    vec_env = EnvWrapper(env_fn)

    # set arguments
    args.device = vec_env.unwrapped.rl_device
    args.n_envs = vec_env.unwrapped.num_envs
    args.max_episode_len = vec_env.unwrapped.max_episode_length
    args.num_stages = vec_env.unwrapped.num_stages
    args.obs_dim = vec_env.unwrapped.num_obs
    args.state_dim = vec_env.unwrapped.num_states - args.num_stages
    args.action_dim = vec_env.unwrapped.num_acts
    args.reward_dim = vec_env.unwrapped.num_rewards
    args.cost_dim = vec_env.unwrapped.num_costs
    args.action_bound_min = -np.ones(args.action_dim)
    args.action_bound_max = np.ones(args.action_dim)
    args.n_steps = algo_cfg['n_steps']
    args.n_total_steps = task_cfg['n_total_steps']
    args.reward_names = task_cfg["env"]["reward_names"]
    args.cost_names = task_cfg["env"]["cost_names"]
    assert len(args.reward_names) == args.reward_dim
    assert len(args.cost_names) == args.cost_dim
    args.history_len = vec_env.unwrapped.history_len
    args.obs_sym_mat = vec_env.unwrapped.obs_sym_mat
    args.state_sym_mat = vec_env.unwrapped.state_sym_mat
    args.joint_sym_mat = vec_env.unwrapped.joint_sym_mat

    # declare agent
    agent_args = deepcopy(args)
    for key in algo_cfg.keys():
        agent_args.__dict__[key] = algo_cfg[key]
    agent = algo_dict[args.algo_name.lower()](agent_args)
    if args.init_from:
        destination = agent.save_dir
        agent.save_dir = args.init_from
        agent.load(args.model_num)
        if args.reset_reward_stats:
            agent.reward_rms.mean.zero_()
            agent.reward_rms.var.fill_(1)
            agent.reward_rms.count.zero_()
        if args.reset_critics:
            agent.reward_critic.initialize()
            agent.cost_critic.initialize()
            agent.reward_critic_optimizer = torch.optim.Adam(agent.reward_critic.parameters(), lr=agent.critic_lr)
            agent.cost_critic_optimizer = torch.optim.Adam(agent.cost_critic.parameters(), lr=agent.critic_lr)
        if args.reset_optimizers:
            agent.actor_lr = float(algo_cfg['actor_lr'])
            agent.critic_lr = float(algo_cfg['critic_lr'])
            agent.actor_optimizer = torch.optim.Adam((p for p in agent.actor.parameters() if p.requires_grad), lr=agent.actor_lr)
            agent.reward_critic_optimizer = torch.optim.Adam(agent.reward_critic.parameters(), lr=agent.critic_lr)
            agent.cost_critic_optimizer = torch.optim.Adam(agent.cost_critic.parameters(), lr=agent.critic_lr)
        agent.save_dir = destination
        Path(destination, 'checkpoint').mkdir(parents=True, exist_ok=True)
        initial_step = 0
        print(f'[train] warm-start: {args.init_from}/checkpoint/model_{args.model_num}.pt -> {destination}', flush=True)
    else:
        initial_step = agent.load(args.model_num)

    # wandb
    if args.wandb:
        wandb.init(project=args.project_name, config=args)
        if args.comment is not None:
            wandb.run.name = f"{args.name}/{args.comment}"
        else:
            wandb.run.name = f"{args.name}"

    # slackbot
    if args.slack:
        slackbot = Slackbot()

    # logger
    log_name_list = deepcopy(agent_args.logging['task_indep'])
    for log_name in agent_args.logging['reward_dep']:
        log_name_list += [f"{log_name}_{reward_name}" for reward_name in args.reward_names]
    for log_name in agent_args.logging['cost_dep']:
        log_name_list += [f"{log_name}_{cost_name}" for cost_name in args.cost_names]
    logger = Logger(log_name_list, f"{args.save_dir}/logs")

    # set train parameters
    reward_sums_tensor = torch.zeros((args.n_envs, args.reward_dim), device=args.device, requires_grad=False, dtype=torch.float32)
    cost_sums_tensor = torch.zeros((args.n_envs, args.cost_dim), device=args.device, requires_grad=False, dtype=torch.float32)
    fail_sums_tensor = torch.zeros(args.n_envs, device=args.device, requires_grad=False, dtype=torch.float32)
    env_cnts_tensor = torch.zeros(args.n_envs, device=args.device, requires_grad=False, dtype=torch.int)
    total_step = initial_step
    wandb_step = initial_step
    slack_step = initial_step
    save_step = initial_step
    console_step = initial_step
    console_started = time.time()
    stage_counts = torch.zeros(args.num_stages, device=args.device)
    episode_count = 0
    failure_count = 0
    full_turn_count = landing_success_count = 0

    # initialize environments
    with torch.no_grad():
        obs_tensor, states_tensor = vec_env.reset()
        stages_tensor = states_tensor[:, -args.num_stages:]
        states_tensor = states_tensor[:, :-args.num_stages]

    # start training
    for _ in range(int(initial_step/args.n_steps), int(args.n_total_steps/args.n_steps)):
        start_time = time.time()

        for _ in range(int(args.n_steps/args.n_envs)):
            env_cnts_tensor += 1
            total_step += args.n_envs

            # ======= collect trajectories & training ======= #
            with torch.no_grad():
                stage_counts += stages_tensor.sum(dim=0)
                actions_tensor = agent.getAction(obs_tensor, states_tensor, stages_tensor, False)
                obs_tensor, states_tensor, rewards_tensor, dones, infos = vec_env.step(actions_tensor)
                stages_tensor = states_tensor[:, -args.num_stages:]
                states_tensor = states_tensor[:, :-args.num_stages]
                next_obs_tensor = infos['next_obs']
                next_stages_tensor = infos['next_states'][:, -args.num_stages:]
                next_states_tensor = infos['next_states'][:, :-args.num_stages]
                costs_tensor = infos['costs']
                fails_tensor = infos['fails']
                dones_tensor = infos['dones']

            reward_sums_tensor += rewards_tensor
            cost_sums_tensor += costs_tensor
            fail_sums_tensor += fails_tensor
            episode_count += int(dones_tensor.sum())
            failure_count += int(fails_tensor.sum())
            if 'one_turn' in infos:
                full_turn_count += int((infos['one_turn'].bool() & dones_tensor.bool()).sum())
                landing_success_count += int((infos['landing_success'].bool() & dones_tensor.bool()
                                              & ~fails_tensor.bool()).sum())

            agent.step(rewards_tensor, costs_tensor, dones_tensor, fails_tensor, 
                       next_obs_tensor, next_states_tensor, next_stages_tensor)
            # =============================================== #

            # wandb logging
            if total_step - wandb_step >= args.wandb_freq and args.wandb:
                wandb_step += args.wandb_freq

                # write log using logger
                env_cnts = env_cnts_tensor.detach().cpu().numpy()
                reward_sums = reward_sums_tensor.detach().cpu().numpy()
                cost_sums = cost_sums_tensor.detach().cpu().numpy()
                fail_sums = fail_sums_tensor.detach().cpu().numpy()
                if 'eplen' in logger.log_name_list: 
                    logger.writes('eplen', np.stack([env_cnts, env_cnts]).T.tolist())
                if 'fail' in logger.log_name_list: 
                    logger.writes('fail', np.stack([env_cnts, fail_sums]).T.tolist())
                for reward_idx in range(args.reward_dim):
                    log_name = f'reward_sum_{args.reward_names[reward_idx]}'
                    if log_name in logger.log_name_list:
                        logger.writes(log_name, np.stack([env_cnts, reward_sums[:, reward_idx]]).T.tolist())
                for cost_idx in range(args.cost_dim):
                    log_name = f'cost_sum_{args.cost_names[cost_idx]}'
                    if log_name in logger.log_name_list: 
                        logger.writes(log_name, np.stack([env_cnts, cost_sums[:, cost_idx]]).T.tolist())
                reward_sums_tensor[:] = 0
                cost_sums_tensor[:] = 0
                fail_sums_tensor[:] = 0
                env_cnts_tensor[:] = 0

                # write log using wandb
                log_data = {"step": total_step}
                print_len = args.n_envs
                print_len2 = int(args.wandb_freq/args.n_steps)
                for reward_idx, reward_name in enumerate(args.reward_names):
                    for log_name in agent_args.logging['reward_dep']:
                        log_data[f'{log_name}/{reward_name}'] = logger.get_avg(f'{log_name}_{reward_name}', print_len if 'sum' in log_name else print_len2)
                for cost_idx, cost_name in enumerate(args.cost_names):
                    for log_name in agent_args.logging['cost_dep']:
                        log_data[f'{log_name}/{cost_name}'] = logger.get_avg(f'{log_name}_{cost_name}', print_len if 'sum' in log_name else print_len2)
                for log_name in agent_args.logging['task_indep']:
                    log_data[f"metric/{log_name}"] = logger.get_avg(log_name, print_len if log_name in ['eplen', 'fail'] else print_len2)
                wandb.log(log_data)
                print(log_data)

            # send slack message
            if total_step - slack_step >= args.slack_freq and args.slack:
                slackbot.sendMsg(f"{args.project_name}\nname: {wandb.run.name}\nsteps: {total_step}\nlog: {log_data}")
                slack_step += args.slack_freq

            # save
            if total_step - save_step >= args.save_freq:
                save_step += args.save_freq
                agent.save(total_step)
                logger.save()

        # train
        if agent.readyToTrain():
            train_results = agent.train()
            for log_name in train_results.keys():
                if log_name in agent_args.logging['task_indep']:
                    logger.write(log_name, [args.n_steps, train_results[log_name]])
                elif log_name in agent_args.logging['reward_dep']:
                    for reward_idx, reward_name in enumerate(args.reward_names):
                        logger.write(f"{log_name}_{reward_name}", [args.n_steps, train_results[log_name][reward_idx]])
                elif log_name in agent_args.logging['cost_dep']:
                    for cost_idx, cost_name in enumerate(args.cost_names):
                        logger.write(f"{log_name}_{cost_name}", [args.n_steps, train_results[log_name][cost_idx]])

        # calculate FPS
        end_time = time.time()
        fps = args.n_steps/(end_time - start_time)
        if 'fps' in logger.log_name_list:
            logger.write('fps', [args.n_steps, fps])
        if args.console_freq > 0 and total_step - console_step >= args.console_freq:
            fractions = (stage_counts / stage_counts.sum().clamp_min(1)).tolist()
            stages_text = ' '.join(f'{name}={value:.1%}' for name, value in zip(task_cfg['env']['stage_names'], fractions))
            print(f'[train] steps={total_step}/{args.n_total_steps} {stages_text} '
                  f'episodes={episode_count} failures={failure_count} '
                  f'full_turns={full_turn_count} stable_landings={landing_success_count} '
                  f'fps={(total_step-console_step)/(time.time()-console_started):.0f}', flush=True)
            console_step = total_step
            console_started = time.time()
            stage_counts.zero_()
            episode_count = failure_count = 0
            full_turn_count = landing_success_count = 0

    # final save
    agent.save(total_step)
    logger.save()

    # terminate
    vec_env.close()


def test(args, task_cfg, algo_cfg):
    require_checkpoint(args.save_dir, args.model_num)
    # create environments
    task_cfg['env']['num_envs'] = 1
    if 'randomize' in task_cfg['env']:
        task_cfg['env']['randomize']['is_randomized'] = False
    task_cfg['env']['history_len'] = algo_cfg['history_len']
    env_fn = lambda: task_dict[task_cfg['name']](
        cfg=task_cfg, rl_device=args.device_name, sim_device=args.device_name, 
        graphics_device_id=0, headless=(not args.render), 
        virtual_screen_capture=False, force_render=args.render
    )
    vec_env = EnvWrapper(env_fn)
    # set arguments
    args.device = vec_env.unwrapped.rl_device
    args.n_envs = vec_env.unwrapped.num_envs
    args.max_episode_len = vec_env.unwrapped.max_episode_length
    args.num_stages = vec_env.unwrapped.num_stages
    args.obs_dim = vec_env.unwrapped.num_obs
    args.state_dim = vec_env.unwrapped.num_states - args.num_stages
    args.action_dim = vec_env.unwrapped.num_acts
    args.reward_dim = vec_env.unwrapped.num_rewards
    args.cost_dim = vec_env.unwrapped.num_costs
    args.action_bound_min = -np.ones(args.action_dim)
    args.action_bound_max = np.ones(args.action_dim)
    args.n_steps = algo_cfg['n_steps']
    args.n_total_steps = task_cfg['n_total_steps']
    args.reward_names = task_cfg["env"]["reward_names"]
    args.cost_names = task_cfg["env"]["cost_names"]
    assert len(args.reward_names) == args.reward_dim
    assert len(args.cost_names) == args.cost_dim
    args.history_len = vec_env.unwrapped.history_len
    args.obs_sym_mat = vec_env.unwrapped.obs_sym_mat
    args.state_sym_mat = vec_env.unwrapped.state_sym_mat
    args.joint_sym_mat = vec_env.unwrapped.joint_sym_mat

    # declare agent
    agent_args = deepcopy(args)
    for key in algo_cfg.keys():
        agent_args.__dict__[key] = algo_cfg[key]
    agent = algo_dict[args.algo_name.lower()](agent_args)
    agent.load(args.model_num)

    with torch.no_grad():
        obs_tensor, states_tensor = vec_env.reset(is_uniform_rollout=False)
        stages_tensor = states_tensor[:, -args.num_stages:]
        states_tensor = states_tensor[:, :-args.num_stages]

    # start rollouts
    for _ in range(100):
        reward_sums_tensor = torch.zeros((args.n_envs, args.reward_dim), device=args.device, requires_grad=False, dtype=torch.float32)
        cost_sums_tensor = torch.zeros((args.n_envs, args.cost_dim), device=args.device, requires_grad=False, dtype=torch.float32)
        start_time = time.time()

        for step_idx in range(args.max_episode_len):
            with torch.no_grad():
                # actions_tensor = agent.getAction(obs_tensor, states_tensor, stages_tensor, False)
                actions_tensor = agent.getAction(obs_tensor, states_tensor, stages_tensor, True)
                obs_tensor, states_tensor, rewards_tensor, dones_tensor, infos = vec_env.step(actions_tensor)
                stages_tensor = states_tensor[:, -args.num_stages:]
                states_tensor = states_tensor[:, :-args.num_stages]
                reward_sums_tensor += rewards_tensor
                cost_sums_tensor += infos['costs']
                if infos['dones'][0]:
                    break
                elapsed_time = time.time() - start_time
                if elapsed_time < (step_idx + 1)*vec_env.unwrapped.control_dt:
                    time.sleep((step_idx + 1)*vec_env.unwrapped.control_dt - elapsed_time)

        print(time.time() - start_time)
        print(reward_sums_tensor[0].cpu().numpy())
        print(cost_sums_tensor[0].cpu().numpy())
        # =============================================== #


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    # ==== processing args ==== #
    # load configuration file
    with open(args.task_cfg_path, 'r') as f:
        task_cfg = YAML().load(f)
    args.task_name = task_cfg['name']
    with open(args.algo_cfg_path, 'r') as f:
        algo_cfg = YAML().load(f)
    args.algo_name = algo_cfg['name']
    args.name = f"{(args.task_name.lower())}_{(args.algo_name.lower())}"
    # save_dir
    args.save_dir = f"results/{args.name}/seed_{args.seed}"
    # device
    if torch.cuda.is_available() and args.device_type == 'gpu':
        device_name = f'cuda:{args.gpu_idx}'
        cprint('[torch] cuda is used.', bold=True, color='cyan')
    else:
        device_name = 'cpu'
        cprint('[torch] cpu is used.', bold=True, color='cyan')
    args.device_name = device_name
    # ========================= #

    if args.test:
        test(args, task_cfg, algo_cfg)
    else:
        train(args, task_cfg, algo_cfg)
