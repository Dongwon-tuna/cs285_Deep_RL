import os
import time

from cs285.agents.pg_agent import PGAgent

import os
import time

import gym
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.action_noise_wrapper import ActionNoiseWrapper

MAX_NVIDEO = 2


def run_training_loop(args):
    logger = Logger(args.logdir)

    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = gym.make(args.env_name, render_mode=None)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # add action noise, if needed
    if args.action_noise_std > 0:
        assert not discrete, f"Cannot use --action_noise_std for discrete environment {args.env_name}"
        env = ActionNoiseWrapper(env, args.seed, args.action_noise_std)

    max_ep_len = args.ep_len or env.spec.max_episode_steps

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if hasattr(env, "model"):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    agent = PGAgent(
        ob_dim,
        ac_dim,
        discrete,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        gamma=args.discount,
        learning_rate=args.learning_rate,
        use_baseline=args.use_baseline,
        use_reward_to_go=args.use_reward_to_go,
        normalize_advantages=args.normalize_advantages,
        baseline_learning_rate=args.baseline_learning_rate,
        baseline_gradient_steps=args.baseline_gradient_steps,
        gae_lambda=args.gae_lambda,
    )

    total_envsteps = 0
    start_time = time.time()

    for itr in range(args.n_iter):
        print(f"\n********** Iteration {itr} ************")
        #breakpoint()
        # TODO: sample `args.batch_size` transitions using utils.sample_trajectories
        # make sure to use `max_ep_len`
        trajs, envsteps_this_batch = utils.sample_trajectories(
                env, agent.actor, args.batch_size, max_ep_len
            )#None, None  # TODO
        total_envsteps += envsteps_this_batch

        # trajs should be a list of dictionaries of NumPy arrays, where each dictionary corresponds to a trajectory.
        # this line converts this into a single dictionary of lists of NumPy arrays.
        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}
        # for k, v in trajs_dict.items():
        #     print(f"{k}: {[arr.shape for arr in v]}")

        # breakpoint()

            # observation: [(12, 4), (38, 4), (29, 4), (18, 4), (20, 4), (34, 4), (20, 4), (21, 4), (15, 4), (23, 4), (19, 4), (26, 4), (14, 4), (13, 4), (12, 4), (25, 4), (15, 4), (10, 4), (24, 4), (20, 4)]
            # image_obs: [(0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,)]
            # reward: [(12,), (38,), (29,), (18,), (20,), (34,), (20,), (21,), (15,), (23,), (19,), (26,), (14,), (13,), (12,), (25,), (15,), (10,), (24,), (20,)]
            # action: [(12,), (38,), (29,), (18,), (20,), (34,), (20,), (21,), (15,), (23,), (19,), (26,), (14,), (13,), (12,), (25,), (15,), (10,), (24,), (20,)]
            # next_observation: [(12, 4), (38, 4), (29, 4), (18, 4), (20, 4), (34, 4), (20, 4), (21, 4), (15, 4), (23, 4), (19, 4), (26, 4), (14, 4), (13, 4), (12, 4), (25, 4), (15, 4), (10, 4), (24, 4), (20, 4)]
            # terminal: [(12,), (38,), (29,), (18,), (20,), (34,), (20,), (21,), (15,), (23,), (19,), (26,), (14,), (13,), (12,), (25,), (15,), (10,), (24,), (20,)]
                # trajs_dict 구조
                # 키(observation, action, reward, next_observation, terminal, image_obs)마다 리스트가 있고,
                # 그 리스트의 i번째 원소가 i번째 trajectory의 전체 NumPy 배열입니다.
                #
                # 배열의 shape 의미
                # observation[i].shape = (T_i, 4) → i번째 에피소드가 T_i 타임스텝 동안 진행되었고, 상태 벡터가 4차원이라는 뜻.
                # reward[i].shape = (T_i,), action[i].shape = (T_i,), terminal[i].shape = (T_i,) → 각 시점별 보상·행동·종료 플래그.
                # image_obs[i].shape = (0,) → render=False라 영상 프레임이 없다는 의미.
                #
                # 독립적인 에피소드
                # (15,4), (48,4) 등 서로 다른 T_i 값들은 서로 다른 두 에피소드의 길이를 나타내며, 이어지는 연속이 아닙니다.
                # 에피소드가 끝나면 환경을 reset하고 새로운 trajectory를 수집합니다.
                #
                # next_observation의 역할
                # observation[t] = s_t 이고,
                # next_observation[t] = s_{t+1} → 상태 s_t에서 행동 a_t을 취한 다음 돌아온 상태.
                #
                # 업데이트 방식 (agent.update)
                # 각 trajectory 리스트를 flatten(쭉 펼쳐서) 하나의 배치(batch)로 합친 뒤,
                # 그 배치 전체에 대해 한 번의 gradient step을 수행해 정책(actor)을 업데이트합니다.
        
        # 각 traj별 observation 값 출력
        # for i, obs in enumerate(trajs_dict["observation"]):
        #     print(f"--- Trajectory {i} Observations (shape={obs.shape}) ---")
        #     print(obs)  # 2D 배열 전체 출력
        #     print()

        # # 각 traj별 reward 값 출력
        # for i, rew in enumerate(trajs_dict["reward"]):
        #     print(f"--- Trajectory {i} Rewards      (shape={rew.shape}) ---")
        #     print(rew)  # 1D 배열 전체 출력
        #     print()
        #     #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
        #     # 보상이 다 1인데 이건 잘 서있으면 1이라는 보상을 주고 넘어지면 done = True 되어서 보상음슴

        # TODO: train the agent using the sampled trajectories and the agent's update function
        train_info: dict = agent.update(
                obs=trajs_dict["observation"],
                actions=trajs_dict["action"],
                rewards=trajs_dict["reward"],
                terminals=trajs_dict["terminal"],
            ) # 이 업데이트는 pg_agent.py에서 해줘야함

        if itr % args.scalar_log_freq == 0:
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
                env, agent.actor, args.eval_batch_size, max_ep_len
            )

            logs = utils.compute_metrics(trajs, eval_trajs)
            # compute additional metrics
            logs.update(train_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs[
                    "Train_AverageReturn"
                ]

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                logger.log_scalar(value, key, itr)
            print("Done logging...\n\n")

            logger.flush()

        if args.video_log_freq != -1 and itr % args.video_log_freq == 0:
            print("\nCollecting video rollouts...")
            eval_video_trajs = utils.sample_n_trajectories(
                env, agent.actor, MAX_NVIDEO, max_ep_len, render=True
            )

            logger.log_trajs_as_videos(
                eval_video_trajs,
                itr,
                fps=fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=64)

    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)

    parser.add_argument("--action_noise_std", type=float, default=0)

    parser.add_argument("--num_eval_trajectories", "-net", type=int, default=10)


    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "q2_pg_"  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = (
        logdir_prefix
        + args.exp_name
        + "_"
        + args.env_name
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join(data_path, logdir)
    args.logdir = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    run_training_loop(args)


if __name__ == "__main__":
    main()
