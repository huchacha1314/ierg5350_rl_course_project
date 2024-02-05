"""
This file implements the train scripts for PPO. You don't need to modify this file.

-----
*2020-2021 Term 1, IERG 5350: Reinforcement Learning. Department of Information Engineering, The Chinese University of
Hong Kong. Course Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao, SUN Hao, ZHAN Xiaohang.*
创建环境 env=KukaPegInsertionGymEnv
创建训练环境 trainer = PPOTrainer(env, config)
获得观察 obs = env.reset()
    训练 _train(trainer, env, eval_env, config, num_envs, algo, log_dir, False, False)
    获取数据 obs, reward, done, info, masks, total_episodes, total_steps, episode_rewards = step_envs(
                    cpu_actions, envs, episode_rewards, reward_recorder, episode_length_recorder, total_steps,
                    total_episodes, config.device
                )
"""
import argparse
import datetime
import os
from collections import deque

import gym
import numpy as np
import torch
# from core.a2c_trainer import A2CTrainer, a2c_config
from PPO.ppo_trainer import PPOTrainer, ppo_config
from PPO.utils import verify_log_dir, pretty_print, Timer, evaluate, \
    summary, save_progress, step_envs
from environments.kuka_peg_env import KukaPegInsertionGymEnv

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gym.logger.set_level(40)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--algo",
    default="PPO",
    type=str,
    help="(Required) The algorithm you want to run. Must in [PPO, A2C]."
)
parser.add_argument(
    "--log-dir",
    default="run/",
    type=str,
    help="The path of directory that you want to store the data to. "
         "Default: data"
)
parser.add_argument(
    "--num-envs",
    default=1,
    type=int,
    help="The number of parallel environments. Default: 15"
)
parser.add_argument(
    "--num-eval-envs",
    default=0,
    type=int,
    help="The number of parallel environments for evaluation. Default: 0 (Do not evaluate)."
)
parser.add_argument(
    "--learning-rate", "-LR",
    default=5e-4,
    type=float,
    help="The learning rate. Default: 5e-4"
)
parser.add_argument(
    "--seed",
    default=100,
    type=int,
    help="The random seed. Default: 100"
)
parser.add_argument(
    "--max-steps",
    "-N",
    default=1e8,
    type=float,
    help="The random seed. Default: 1e7"
)
parser.add_argument(
    "--lr",
    default=5e-4,
    type=float,
    help="The learning rate. Default: 5e-4"
)
parser.add_argument(
    "--test",
    action="store_true",
    help="If test, then disable multiprocessing."
)
parser.add_argument(
    "--action-repeat",
    default=1,
    type=int,
    help="If action_repeat>0, then set action_repeat. For cCarRacing-v0 only."
)
parser.add_argument(
    "--entropy",
    default=0.01,
    type=float,
    help="Set entropy_loss_weight for training."
)
parser.add_argument(
    "--env-id",
    default="cPong-v0",
    type=str,
    choices=["cPong-v0", "cCarRacing-v0"]
)
parser.add_argument(
    "--restore",
    default="",
    type=str,
    help="Restore your agent if you wish. Must in this format: '...xxxx/checkpoint-yyyy.pkl'."
)
parser.add_argument(
    "--opponent",
    default="",
    type=str,
    help="Restore an agent as opponent and conduct two-player game. Must ended with '...xxxx/checkpoint-yyyy.pkl'"
)
args = parser.parse_args()


def train(args):
    # Verify algorithm and config 设置训练超参数
    algo = args.algo
    if algo == "PPO":
        config = ppo_config
    else:
        raise ValueError("args.algo must in [PPO]")
    config.num_envs = args.num_envs
    config.lr = args.lr
    config.entropy_loss_weight = args.entropy
    assert args.env_id in ["cPong-v0", "cCarRacing-v0"], args.env_id

    # Seed the environments and setup torch 设置随机种子 并且初始化环境
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)

    # Create vectorized environments 创建并行环境
    num_envs = args.num_envs
    env_id = 0

    # Clean log directory
    log_dir = verify_log_dir(args.log_dir, "{}_{}_{}".format(
        env_id, algo, datetime.datetime.now().strftime("%m-%d_%H-%M")
    ))
    
    #创建环境
    env = KukaPegInsertionGymEnv(renders=False,
                                 srl_model="raw_pixels",
                                 is_discrete=False,
                                 shape_reward=False,
                                 force_down=False)

    eval_env = env

    # Setup trainer
    if algo == "PPO":
        trainer = PPOTrainer(env, config)
    else:
        raise ValueError("Unknown algorithm {}".format(algo))


    #如果 --restore 加载预训练模型
    if args.restore:
        restore_log_dir = os.path.dirname(args.restore)
        restore_suffix = os.path.basename(args.restore).split("checkpoint-")[1].split(".pkl")[0]
        success = trainer.load_w(restore_log_dir, restore_suffix)
        if not success:
            raise ValueError("We can't restore your agent. The log_dir is {} and the suffix is {}".format(
                restore_log_dir, restore_suffix
            ))

    # Start training
    print("Start training!")
    #重新加载环境，并返回初始观察obs
    #通常’reset‘方法在每个episode 的开始调用，将环境重置为初始状态并且获取初始观察
    obs = env.reset()
    # frame_stack_tensor.update(obs)
    #为新的迭代准备训练的数据结构
    trainer.rollouts.before_update(obs)

    try:
        _train(trainer, env, eval_env, config, num_envs, algo, log_dir, False, False)
    except KeyboardInterrupt:
        print("The training is stopped by user. The log directory is {}. Now we finish the training.".format(log_dir))

    trainer.save_w(log_dir, "final")
    env.close()


def _train(trainer, envs, eval_envs, config, num_envs, algo, log_dir, tournament, test):
    # Setup some stats helpers
    #设置了统计的辅助变量
    episode_rewards = np.zeros([num_envs, 1], dtype=np.float)
    total_episodes = total_steps = iteration = 0
    reward_recorder = deque(maxlen=100)
    episode_length_recorder = deque(maxlen=100)
    sample_timer = Timer()
    process_timer = Timer()
    update_timer = Timer()
    total_timer = Timer()
    progress = []
    evaluate_stat = {}

    #当总步数超过最大值的时候跳出循环
    while True:  # Break when total_steps exceeds maximum value
        # ===== Sample Data =====
        #采样数据
        # with 在python 中常常和计时器一起工作，用于测量缩紧内的运行时间，进入缩进代码的时候开始计时，退出时结束计时
        with sample_timer:
            for index in range(config.num_steps):
                # Get action
                # with 在python 中和上下文管理器一起工作，在进入缩紧程序之前获取数据，在结束缩进程序之后退出，因为在训练阶段 不会进行梯度下降工作
                with torch.no_grad():
                    values, actions, action_log_prob = trainer.compute_action(trainer.rollouts.get_observation(index))

                if trainer.discrete:
                    cpu_actions = actions.view(-1).cpu().numpy()
                else:
                    cpu_actions = actions.squeeze(0).cpu().numpy()

                # Step the environment
                # (Check step_envs function, you need to implement it)
                #执行环境步骤
                obs, reward, done, info, masks, total_episodes, total_steps, episode_rewards = step_envs(
                    cpu_actions, envs, episode_rewards, reward_recorder, episode_length_recorder, total_steps,
                    total_episodes, config.device
                )

                reward = np.array([reward])
                rewards = torch.from_numpy(
                    reward.astype(np.float32)).view(-1, 1).to(config.device)

                # Store samples
                #存储样本
                if trainer.discrete:
                    actions = actions.view(-1, 1)

                trainer.rollouts.insert(obs, actions, action_log_prob, values, rewards, masks)

                # # break the loop and reset the env
                if done:
                    obs = envs.reset()
                    trainer.rollouts.before_update(obs)
                    break

        # ===== Process Samples =====
        #处理样本
        with process_timer:
            with torch.no_grad():
                next_value = trainer.compute_values(trainer.rollouts.get_observation(-1))
            trainer.rollouts.compute_returns(next_value, config.gamma)

        # ===== Update Policy =====
        #更新策略
        with update_timer:
            policy_loss, value_loss, dist_entropy, total_loss = trainer.update(trainer.rollouts)
            trainer.rollouts.after_update()

        # ===== Evaluate Current Policy =====
        #评估策略
        if eval_envs is not None and iteration % config.eval_freq == 0:
            eval_timer = Timer()
            evaluate_rewards, evaluate_lengths = evaluate(trainer, eval_envs, 20)
            evaluate_stat = summary(evaluate_rewards, "episode_reward")
            if evaluate_lengths:
                evaluate_stat.update(summary(evaluate_lengths, "episode_length"))
            evaluate_stat.update(dict(
                win_rate=float(
                    sum(np.array(evaluate_rewards) >= 0) / len(
                        evaluate_rewards)),
                evaluate_time=eval_timer.now,
                evaluate_iteration=iteration
            ))

        # ===== Log information =====
        #记录信息
        if iteration % config.log_freq == 0:
            stats = dict(
                log_dir=log_dir,
                frame_per_second=int(total_steps / total_timer.now),
                training_episode_reward=summary(reward_recorder, "episode_reward"),
                training_episode_length=summary(episode_length_recorder, "episode_length"),
                evaluate_stats=evaluate_stat,
                learning_stats=dict(
                    policy_loss=policy_loss,
                    entropy=dist_entropy,
                    value_loss=value_loss,
                    total_loss=total_loss
                ),
                total_steps=total_steps,
                total_episodes=total_episodes,
                time_stats=dict(
                    sample_time=sample_timer.avg,
                    process_time=process_timer.avg,
                    update_time=update_timer.avg,
                    total_time=total_timer.now,
                    episode_time=sample_timer.avg + process_timer.avg + update_timer.avg
                ),
                iteration=iteration
            )

            progress.append(stats)

            from IPython.display import clear_output
            clear_output()

            pretty_print({
                "===== {} Training Iteration {} =====".format(
                    algo, iteration): stats
            })
            progress_path = save_progress(log_dir, progress)

        #跳出循环

        if iteration % config.save_freq == 0:
            trainer_path = trainer.save_w(log_dir, "iter{}".format(iteration))
            progress_path = save_progress(log_dir, progress)
            print("Saved trainer state at <{}>. Saved progress at <{}>.".format(
                trainer_path, progress_path
            ))

        if total_steps > int(args.max_steps):
            break

        iteration += 1


if __name__ == '__main__':
    train(args)
