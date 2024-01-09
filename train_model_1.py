import gymnasium as gym
from stable_baselines3 import DDPG, PPO
import os
import re
import time

## --- DDPG ---
def train_model_DDPG(env_name,
                    num_timesteps, gravity, enable_wind, wind_power, turbulence_power, 
                    file_ver, folder_ver, index, folder_path, file_path):

    start_time = time.time()
    env = gym.make(env_name, gravity=gravity, enable_wind=enable_wind,
                   wind_power=wind_power, turbulence_power=turbulence_power,
                   continuous = True)
                   #render_mode="human")
## DDPG model set
    model = DDPG("MlpPolicy", env, verbose=1, 
                tensorboard_log=(folder_path + "DDPG_" + str(folder_ver) + "_" +str(index)
                                  + "_grav" + str(float(gravity)) + "_wnd" + str(float(wind_power))
                                    + "_turb" + str(float(turbulence_power)))) # => Launch Tensorboard on VS Code
## model learn
    model.learn(total_timesteps=num_timesteps)
## DDPG model save
    model.save(file_path + "ddpg_lunar_lander_"+str(file_ver)+"_"+str(index)
                                  + "_grav" + str(float(gravity)) + "_wnd" + str(float(wind_power))
                                    + "_turb" + str(float(turbulence_power)))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

    fldr_pth = (folder_path + "DDPG_" + str(folder_ver) + "_" +str(index)
                                  + "_grav" + str(float(gravity)) + "_wnd" + str(float(wind_power))
                                    + "_turb" + str(float(turbulence_power)))
# Create a .txt file together training data and attach it next to it: 
## DDPG .txt file
    with open(fldr_pth +"/"+"info_DDPG_" + str(folder_ver) + "_" +str(index)+".txt", "w") as file:
        
        file.write(f"\
--------INFO--------\n\n\
        DDPG\n\
number of time steps =  {num_timesteps}\n\
gravity              =  {gravity}\n\
enable_wind          =  {enable_wind}\n\
wind_power           =  {wind_power}\n\
turbulence_power     =  {turbulence_power}\n\n\
--  --  --  --  --  --  --  --  --  --  --  --\n\
Elapsed time during training = {elapsed_time}")

## --- PPO ---
def train_model_PPO(env_name, 
                    num_timesteps, gravity, enable_wind, wind_power, turbulence_power, 
                    file_ver, folder_ver, index, folder_path, file_path):

    start_time = time.time()  
    env = gym.make(env_name, gravity=gravity, enable_wind=enable_wind,
                   wind_power=wind_power, turbulence_power=turbulence_power,
                   continuous = True)
                   #render_mode="human")
## PPO model
    model = PPO("MlpPolicy", env, verbose=1,
                 tensorboard_log=(folder_path + "PPO_" + str(folder_ver) + "_" +str(index)
                                  + "_grav" + str(float(gravity)) + "_wnd" + str(float(wind_power))
                                    + "_turb" + str(float(turbulence_power)))) # => Launch Tensorboard on VS Code
## model learn    
    model.learn(total_timesteps=num_timesteps)

## PPO model save
    model.save(file_path + "ppo_lunar_lander_"+str(file_ver)+"_"+str(index)
                                  + "_grav" + str(float(gravity)) + "_wnd" + str(float(wind_power))
                                    + "_turb" + str(float(turbulence_power)))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

    fldr_pth = (folder_path + "PPO_" + str(folder_ver) + "_" +str(index)
                                  + "_grav" + str(float(gravity)) + "_wnd" + str(float(wind_power))
                                    + "_turb" + str(float(turbulence_power)))

    # Create a .txt file together training data and attach it next to it:   
## PPO .txt file
    with open(fldr_pth +"/"+ "info_PPO_" + str(folder_ver) + "_" +str(index)+".txt", "w") as file:
       
        file.write(f"\
--------INFO--------\n\n\
        PPO\n\
number of time steps =  {num_timesteps}\n\
gravity              =  {gravity}\n\
enable_wind          =  {enable_wind}\n\
wind_power           =  {wind_power}\n\
turbulence_power     =  {turbulence_power}\n\n\
--  --  --  --  --  --  --  --  --  --  --  \n\
Elapsed time during training = {elapsed_time}")
        
def latest_version(folder_path):

    # Get ready latest version number for iterative search
    latest_ver = 0 
    # If no file or folder exists return 0 as version
    if not os.path.exists(folder_path):
        return latest_ver
    else:            
        # Iterate over the contents of the parent folder
        for f_name in os.listdir(folder_path):

            # find all the numeric elements in the name of file or the folder with reg. ex. search
            folder_components = str(re.findall(r"\d+", f_name)[:1]) 
            # folder_name = str(folder_components[:1])  # str(folder_components[:1][0:])
            # The found number list consist of [' '] elements. Get rid of them
            folder_name = folder_components.translate(str.maketrans("", "", "[]'"))

            # Check if the folder version name is a valid number
            if folder_name.isnumeric():
                # Extract the folder number
                folder_number = int(folder_name)
                # Update the highest number if it's greater than the current highest
                if folder_number >= latest_ver:
                    latest_ver = folder_number
    return latest_ver


def train_model(num_timesteps, gravity, enable_wind, wind_power, turbulence_power, loop_2times, which_model):

    main_path = os.path.dirname(__file__)+"/lunarlander_"
    folder_path = main_path+"tensorboard/"
    file_path = main_path+"training/"
    # Find and set up-dated version numbers. Get to next number for the training
    folder_ver = latest_version(folder_path)+1
    file_ver = latest_version(file_path)+1

    if which_model == 'PPO':
        if loop_2times == False:
            train_model_PPO("LunarLanderContinuous-v2",
            num_timesteps = num_timesteps, gravity = gravity, enable_wind = enable_wind,
            wind_power = wind_power, turbulence_power = turbulence_power, 
            # saved trainings will be held in distict folders ..._ver_indx            
            folder_ver=folder_ver, file_ver=file_ver, index=1,
            folder_path=folder_path, file_path = file_path) # assign to set file and folder name automatically.

        if loop_2times == True:
            
            # loop each training for 3 times for measuring the differences
            for i in range(2):
                train_model_PPO("LunarLanderContinuous-v2",
                num_timesteps = num_timesteps, gravity = gravity, enable_wind = enable_wind,
                wind_power = wind_power, turbulence_power = turbulence_power, 
                # saved trainings will be held in distict folders ..._ver_indx            
                folder_ver=folder_ver, file_ver=file_ver, index=i, 
                folder_path=folder_path, file_path = file_path) # assign to set file and folder name automatically.
    else:
        if loop_2times == False:
            train_model_DDPG("LunarLanderContinuous-v2",
            num_timesteps = num_timesteps, gravity = gravity, enable_wind = enable_wind,
            wind_power = wind_power, turbulence_power = turbulence_power, 
            # saved trainings will be held in distict folders ..._ver_indx            
            folder_ver=folder_ver, file_ver=file_ver, index=1, 
            folder_path=folder_path, file_path = file_path) # assign to set file and folder name automatically.

        if loop_2times == True:
            
            # loop each training for 3 times for measuring the differences
            for i in range(2):
                train_model_DDPG("LunarLanderContinuous-v2",
                num_timesteps = num_timesteps, gravity = gravity, enable_wind = enable_wind,
                wind_power = wind_power, turbulence_power = turbulence_power, 
                # saved trainings will be held in distict folders ..._ver_indx            
                folder_ver=folder_ver, file_ver=file_ver, index=i, 
                folder_path=folder_path, file_path = file_path) # assign to set file and folder name automatically.

if __name__ == "__main__":

    num_timesteps = 750000
    gravity=-1.0
    enable_wind=True
    wind_power=20
    turbulence_power=0
    # Is the training repeated for 2 times
    loop_2times = False
    which_model = 'PPO' # 'PPO' or 'DDPG'
    
    train_model(num_timesteps = num_timesteps, gravity = gravity,
         enable_wind = enable_wind, wind_power = wind_power, turbulence_power = turbulence_power,
         loop_2times = loop_2times, which_model = which_model)
    
    gravity=-1.2
    
    train_model(num_timesteps = num_timesteps, gravity = gravity,
         enable_wind = enable_wind, wind_power = wind_power, turbulence_power = turbulence_power,
         loop_2times = loop_2times, which_model = which_model)

    gravity=-1.4
    
    train_model(num_timesteps = num_timesteps, gravity = gravity,
         enable_wind = enable_wind, wind_power = wind_power, turbulence_power = turbulence_power,
         loop_2times = loop_2times, which_model = which_model)        

    gravity=-1.6

    train_model(num_timesteps = num_timesteps, gravity = gravity,
         enable_wind = enable_wind, wind_power = wind_power, turbulence_power = turbulence_power,
         loop_2times = loop_2times, which_model = which_model)

    gravity=-1.8

    train_model(num_timesteps = num_timesteps, gravity = gravity,
         enable_wind = enable_wind, wind_power = wind_power, turbulence_power = turbulence_power,
         loop_2times = loop_2times, which_model = which_model)