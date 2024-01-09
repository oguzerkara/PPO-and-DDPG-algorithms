import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np


# https://stable-baselines.readthedocs.io/en/master/guide/tensorboard.html?highlight=TensorBoardCallback#logging-more-values

# For  checking git bug issues

env = gym.make("LunarLanderContinuous-v2",
               gravity=-10.0,
               enable_wind=False,
               turbulence_power=1.5,
               render_mode="human")
env = DummyVecEnv([lambda: env])


def evaluate_model(model_path, num_eval_episodes=10):
    trained_model = DDPG.load(model_path)

    eval_env = Monitor(env, filename=None)
    mean_reward, std_reward = evaluate_policy(trained_model, eval_env, n_eval_episodes=num_eval_episodes)

    print(f"Mean reward: {mean_reward}")
    print(f"Standard deviation of reward: {std_reward}")


def visualize_trained_agent(model_path, env):
    trained_model = DDPG.load(model_path)
    obs = env.reset()
    for _ in range(10000):
        action, _states = trained_model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render("human")


# Close the environment
env.close()

if __name__ == "__main__":
    # evaluate_model("ddpg_lunar_lander")
    visualize_trained_agent("/Users/atamerkara/backup_datasets_tum_adlr_ws24_11/18.12_10.45am_lunarlander_training/ddpg_lunar_lander_7_1.zip", env)