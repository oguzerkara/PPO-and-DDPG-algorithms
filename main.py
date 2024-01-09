from evaluate_model import evaluate_model
from train_model_1 import train_model

if __name__ == "__main__":
    #train_model("LunarLanderContinuous-v2", num_timesteps=1000000, save_path="ddpg_lunar_lander")
    evaluate_model("ddpg_lunar_lander")