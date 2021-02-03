from stable_baselines import PPO2, logger
from stable_baselines.common.cmd_util import make_atari_env, atari_arg_parser
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
from datetime import datetime, timezone
from pathlib import Path

def train(env_id, num_timesteps, seed, policy, load_addr=None, save_addr="ppo2_model", n_envs=8, nminibatches=4,
          n_steps=128, logdir=None, full_logs=False, timesteps_per_save=None):
    """
    Train PPO2 model for atari environment, for testing purposes

    :param env_id: (str) the environment id string
    :param num_timesteps: (int) the number of timesteps to run
    :param seed: (int) Used to seed the random generator.
    :param policy: (Object) The policy model to use (MLP, CNN, LSTM, ...)
    :param n_envs: (int) Number of parallel environments
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches.
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    """

    env = VecFrameStack(make_atari_env(env_id, n_envs, seed), 4)
    policy = {'cnn': CnnPolicy, 'lstm': CnnLstmPolicy, 'lnlstm': CnnLnLstmPolicy, 'mlp': MlpPolicy}[policy]
    if load_addr:
        model = PPO2.load(load_addr, env=env)
    else:
        model = PPO2(policy=policy, env=env, n_steps=n_steps, nminibatches=nminibatches,
                 lam=0.95, gamma=0.99, noptepochs=4, ent_coef=.01,
                 learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1, verbose=1,
                 full_tensorboard_log=logdir if full_logs else False, tensorboard_log=logdir)

    timesteps_per_save = timesteps_per_save or (num_timesteps // 2)
    total_rounds = num_timesteps // timesteps_per_save
    elapsed_timesteps = 0
    start_time = f'{datetime.now(timezone.utc).strftime("%Y-%m-%d--%H-%M-%S-%Z")}'

    for i in range(total_rounds + 1):
        timesteps_to_do = min(timesteps_per_save, num_timesteps - elapsed_timesteps)
        if timesteps_to_do > 0:
            model.learn(total_timesteps=timesteps_to_do, reset_num_timesteps=False, target_timesteps=num_timesteps)
                        # elapsed_timesteps=elapsed_timesteps)
            elapsed_timesteps += timesteps_to_do
            pathdir = f"{save_addr}_{start_time}_{elapsed_timesteps}"
            Path(pathdir).mkdir(parents=True, exist_ok=True)
            model.save(pathdir)
    env.close()
    # Free memory
    del model


def main():
    """
    Runs the test
    """
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='cnn')
    parser.add_argument('--logdir', help='Directory to save Tensorboard logs', default=None)
    parser.add_argument('--load_addr', help='Directory to load model', default=None)
    parser.add_argument('--save_addr', help='Directory to save model', default="ppo2_breakout")
    parser.add_argument('--timesteps_per_save', help='', type=int, default=None)
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, load_addr=args.load_addr, full_logs=True,
          save_addr=args.save_addr, policy=args.policy, logdir=args.logdir, timesteps_per_save=args.timesteps_per_save)


if __name__ == '__main__':
    main()
