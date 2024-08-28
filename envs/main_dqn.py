import numpy as np
from rlss_envs import ServerlessEnv 
from dqn_agent import Agent as dqn
from gymnasium import spaces
import matplotlib.pyplot as plt
import argparse
import torch
import json
import os
from datetime import datetime
import time
import re

now = datetime.now()
folder_base = f"result/result_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"

def plot_log_fig(log_folder):
    log_file = os.path.join(log_folder, 'log.txt')

    # Đọc file log
    with open(log_file, 'r') as file:
        log_data = file.read()
    
    training_num_pattern = re.compile(r'Test trainned model (\d+) times') 
    training_num_match = training_num_pattern.search(log_data)   
    training_num = int(training_num_match.group(1))
    
    
    # Lấy giá trị timestep từ ENVIRONMENT PARAMETERS
    timestep_pattern = re.compile(r'"timestep": (\d+),')
    timestep_match = timestep_pattern.search(log_data)
    if timestep_match:
        timestep_value = int(timestep_match.group(1))

    # Tìm các khối "SYSTEM EVALUATION PARAMETERS IN TIMESTEP"
    timestep_blocks = re.findall(r'SYSTEM EVALUATION PARAMETERS IN TIMESTEP \d+:.*?(?=SYSTEM EVALUATION PARAMETERS IN TIMESTEP \d+:|\Z)', log_data, re.S)

    # Khởi tạo các danh sách để lưu trữ giá trị
    new_rqs = []
    in_queue_rqs = []
    in_sys_rqs = []
    done_rqs = []
    rewards = []
    energy_consumptions = []

    # Trích xuất các thông số từ mỗi khối
    for i, block in enumerate(timestep_blocks, start=1):
        new_rq_match = re.search(r'Number new request\s*:\s*(\d+)', block)
        new_rq = int(new_rq_match.group(1)) if new_rq_match else None
        
        in_queue_rq_match = re.search(r'Number in queue request\s*:\s*(\d+)', block)
        in_queue_rq = int(in_queue_rq_match.group(1)) if in_queue_rq_match else None
        
        in_sys_rq_match = re.search(r'Number in system request\s*:\s*(\d+)', block)
        in_sys_rq = int(in_sys_rq_match.group(1)) if in_sys_rq_match else None
        
        # Tìm Cumulative number accepted request
        done_rq_match = re.search(r'Number done system request\s*:\s*(\d+)', block)
        done_rq = int(done_rq_match.group(1)) if done_rq_match else None
        
        # Tìm Rewards
        rewards_match = re.search(r'Rewards\s*:\s*([-\d.]+)', block)
        reward = float(rewards_match.group(1)) if rewards_match else None
        
        # Tìm Energy Consumption
        energy_match = re.search(r'Energy Consumption\s*:\s*([\d.]+)J', block)
        energy_consumption = float(energy_match.group(1)) if energy_match else None
        
        # Lưu các thông số vào danh sách
        new_rqs.append(new_rq)
        in_queue_rqs.append(in_queue_rq)
        in_sys_rqs.append(in_sys_rq)
        done_rqs.append(done_rq)
        rewards.append(reward)
        energy_consumptions.append(energy_consumption)

    # Tính toán accepted ratio
    acceptance_ratio = [
        (in_sys_rqs[i] + done_rqs[i] - (in_sys_rqs[i-1] if i > 0 else 0)) /
        ((in_queue_rqs[i-1] if i > 0 else 0) + new_rqs[i])
        for i in range(len(new_rqs))
    ]
    
    energy_consumptions = np.array(energy_consumptions)
    acceptance_ratio = np.array(acceptance_ratio)
    rewards = np.array(rewards)

    num_step = len(acceptance_ratio) // training_num
    
    avg_energy_consumptions = [
        np.mean(energy_consumptions[i::num_step]) 
        for i in range(num_step)
    ]

    avg_acceptance_ratio = [
        np.mean(acceptance_ratio[i::num_step]) 
        for i in range(num_step)
    ]
    
    avg_rewards = [
        np.mean(rewards[i::num_step]) 
        for i in range(num_step)
    ]
    
    timesteps = np.arange(len(avg_energy_consumptions)) * timestep_value
    
    # Vẽ accepted ratio trung bình theo thời gian
    plt.figure()
    plt.plot(timesteps, avg_acceptance_ratio, label='Avg Acceptance Ratio Over {} Episodes'.format(training_num), color='purple')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceptance Ratio')
    plt.title('Avg Acceptance Ratio Over {} Episodes'.format(training_num))
    plt.grid(True)
    plt.savefig(os.path.join(log_folder, 'acceptance_ratio_plot.png'))  # Lưu hình

    # Hình Rewards
    plt.figure()
    plt.plot(timesteps, avg_rewards, label='Avg Rewards Over {} Episodes'.format(training_num), color='red')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Rewards')
    plt.title('Avg Rewards Over {} Episodes'.format(training_num))
    plt.grid(True)
    plt.savefig(os.path.join(log_folder, 'rewards_plot.png'))  # Lưu hình

    # Hình Energy Consumption
    plt.figure()
    plt.plot(timesteps, avg_energy_consumptions, label='Avg Energy Consumption Over {} Episodes'.format(training_num), color='orange')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Energy Consumption (J)')
    plt.title('Avg Energy Consumption Over {} Episodes'.format(training_num))
    plt.grid(True)
    plt.savefig(os.path.join(log_folder, 'energy_consumption_plot.png'))  # Lưu hình

    
# testing the trained model
def test(args, env_config, drl_hyper_params):
    if args['folder'] is not None:
        folder_base =  args['folder']
        
    folder_name = os.path.join(folder_base, 'test')
    os.makedirs(folder_name, exist_ok=True)
        
    env_config["log_path"] = os.path.join(folder_name, 'log.txt')
    env = ServerlessEnv(env_config=env_config)
    action_size = env.action_size
    state_dim = env.state_space.shape[0]
    
    
    drl_hyper_params["action_size"] = action_size
    drl_hyper_params["state_dim"] = state_dim          
        

    agent = dqn(drl_hyper_params["state_dim"],
                drl_hyper_params["action_size"], 
                drl_hyper_params["replay_buffer_size"],
                drl_hyper_params["batch_size"],
                drl_hyper_params["hidden_size"],
                drl_hyper_params["gamma"],
                drl_hyper_params["learning_rate"],
                folder_base)
    
    # Loading the Model's weights generated by the selected model
    agent.load_models()

    if args['observe'] is None:
        eps = 1
    else:
        eps = int(args['observe'])  
      
    with open(env_config["log_path"], 'w') as f:
        f.write("Test trainned model {} times\n\n".format(eps))
        f.write("ENVIRONMENT PARAMETERS:\n")
        json.dump(env_config, f, indent=4)
        f.write("\n \n")
        f.write("DRL HYPERPARAMETERS:\n")
        json.dump(drl_hyper_params, f, indent=4)
        f.write("\n \n")
                 
    for e in range(eps):
        done = False
        sub_episode = 0
        cum_reward = 0
        rewards = []
        state = env.reset()
        state = np.reshape(state, [state_dim])
        while not done:
            action = agent.get_action(state, env=env, epsilon=0)
            next_state, reward, done, _ = env.step(action)
            env.render()
            next_state = np.reshape(next_state, [state_dim])
            state = next_state
            rewards.append(reward)
            cum_reward += reward
            sub_episode += 1
        with open(env_config["log_path"], 'a') as f:
            f.write("\n********************************************************************************\n")
            f.write("Sub_episode: {}, reward: {}".format(sub_episode, cum_reward))
            f.write("\n********************************************************************************\n\n\n")
    
    plot_log_fig(folder_name)

            
# Training the model            
def train(args, env_config, drl_hyper_params):
    folder_name = os.path.join(folder_base, 'train')
    os.makedirs(folder_name, exist_ok=True)
    
    env_config["log_path"] = os.path.join(folder_name, 'log.txt')
    env = ServerlessEnv(env_config=env_config)
    action_size = env.action_size
    state_dim = env.state_space.shape[0]
    
    drl_hyper_params["action_size"] = action_size
    drl_hyper_params["state_dim"] = state_dim    
    
    with open(env_config["log_path"], 'w') as f:
        f.write("ENVIRONMENT PARAMETERS:\n")
        json.dump(env_config, f, indent=4)
        f.write("\n \n")
        f.write("DRL HYPERPARAMETERS:\n")
        json.dump(drl_hyper_params, f, indent=4)
        f.write("\n \n")

    # Instantiating the DQN_Agent
    agent = dqn(drl_hyper_params["state_dim"],
                drl_hyper_params["action_size"], 
                drl_hyper_params["replay_buffer_size"],
                drl_hyper_params["batch_size"],
                drl_hyper_params["hidden_size"],
                drl_hyper_params["gamma"],
                drl_hyper_params["learning_rate"],
                folder_base)

    tab = {}
    avg_reward_list = []
    rewards = []
    cumulative_rewards = []
    
    for e in range(drl_hyper_params["episodes"]):
        state = env.reset()
        state = np.reshape(state, [state_dim])
        done = False
        cum_reward = 0
        sub_episode = 0
        while not done:
            action = agent.get_action(state=state,env=env,epsilon=drl_hyper_params["epsilon"])
            next_state, reward, done, _ = env.step(action)
            env.render()
            tab[e * env.current_time + env.current_time] = {"action": action, "reward": reward, "next_state": next_state}
            next_state = np.reshape(next_state, [state_dim])
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            agent.learn()
            rewards.append(reward)
            cum_reward += reward
            sub_episode += 1
            
        # eps = max(eps_end, eps_decay*eps)
        # Saving the Model's weights generated by the selected model
        if e % drl_hyper_params["batch_update"] == 0:
            agent.update_target_network()
            # Saving the Model's weights generated by the selected model
            agent.save_models()
        cumulative_rewards.append(cum_reward)
        with open(env_config["log_path"], 'a') as f:
            f.write("\n********************************************************************************\n")
            f.write("Episode: {}/{}, sub_episode: {}, reward: {}".format(e, drl_hyper_params["episodes"], sub_episode, cum_reward))
            f.write("\n********************************************************************************\n\n\n")
        if e > drl_hyper_params["max_env_steps"]:
            avg = np.mean(cumulative_rewards[-drl_hyper_params["max_env_steps"]:])
        else:
            avg = np.mean(cumulative_rewards)
        avg_reward_list.append(avg)
        plt.figure(2)
        plt.clf()
        rewards_t = torch.tensor(cumulative_rewards, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative reward')
        plt.grid(True)
        plt.plot(rewards_t.numpy())
        # Take max_env_steps episode averages and plot them too
        if len(rewards_t) >= drl_hyper_params["max_env_steps"]:
            means = rewards_t.unfold(0, drl_hyper_params["max_env_steps"], 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(drl_hyper_params["max_env_steps"]-1), means))
            plt.plot(means.numpy())
        # plt.pause(0.001)  # pause a bit so that plots are updated
        plt.savefig(os.path.join(folder_name,'live_average_re0wards_DQN.png'))
        plt.close()
        agent.save_models()
        
    # Plotting the reward/avg_reward
    if args['train'] is not None:
        plt.plot((np.arange(len(avg_reward_list)) + 1), avg_reward_list)
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.title('Average Reward vs Episodes')
        plt.savefig(os.path.join(folder_name, 'average_rewards_{}.png'.format(args['train'])))
        plt.close()

        plt.plot(cumulative_rewards)
        plt.plot(avg_reward_list)
        plt.legend(["Reward", "{}-episode average".format(drl_hyper_params["max_env_steps"])])
        plt.title("Reward history")
        plt.savefig(os.path.join(folder_name, 'Live_average_rewards_{}.png'.format(args['train'])))
        plt.close()

        # Saving all sort of statistics
        with open(os.path.join(folder_name, 'action_state_information_{}.txt'.format(args['train'])), "a") as w:
            w.write(str(tab))
        with open(os.path.join(folder_name, 'detailed_action_selection_{}.txt'.format(args['train'])), "a") as w:
            w.write(str(agent.action))
            
    # with open(os.path.join(folder_name, 'reward_list_{}.txt'.format(args['train'])), "w") as w:
    #     w.write(str(rewards))
    with open(os.path.join(folder_name, 'reward_list_{}.txt'.format(args['train'])), "w") as w:
        w.write(str(cumulative_rewards))
    with open(os.path.join(folder_name, 'average_reward_list_{}.txt'.format(args['train'])), "w") as w:
        w.write(str(avg_reward_list))    

def main(args):
    # Environment variable
    num_service = 1
    timestep = 120
    num_container = [100]
    container_lifetime = 3600*8
    rq_timeout = 20
    average_requests = 160/60
    max_rq_active_time = {"type": "random", "value": 60}
    energy_price = 10e-8 
    ram_profit = 10e-5
    cpu_profit = 10e-5
    alpharw = 0.05
    betarw = 0.05
    gammarw = 0.9 


    # DQN_agent
    episodes = 2000                        # Total episodes for the training
    batch_size = 32                        # Total used memory in memory replay mode
    max_env_steps = 50                    # Max steps per episode
    batch_update = 20
    
    replay_buffer_size=50000
    hidden_size=400
    gamma=0.1
    learning_rate=5e-4
    eps = 0.05
    
    # # Exploration initiation
    # eps = 1.
    # eps_end = 0.01
    # eps_decay = 0.995

    env_config = {"render_mode":None, 
                  "num_service": num_service,
                  "timestep": timestep,
                  "num_container": num_container,
                  "container_lifetime": container_lifetime,
                  "rq_timeout": rq_timeout,
                  "average_requests": average_requests,
                  "max_rq_active_time": max_rq_active_time,
                  "energy_price": energy_price, 
                  "ram_profit": ram_profit,
                  "cpu_profit": cpu_profit,
                  "alpha": alpharw,
                  "beta": betarw,
                  "gamma": gammarw}
    
    drl_hyper_params = {"episodes": episodes,                       
                        "batch_size" :batch_size,                        
                        "max_env_steps": max_env_steps,                    
                        "batch_update" : batch_update,
                        "replay_buffer_size": replay_buffer_size,
                        "hidden_size": hidden_size,
                        "gamma": gamma,
                        "epsilon": eps,
                        "learning_rate":learning_rate}
    

    if args['observe'] is not None:
        test(args, env_config, drl_hyper_params)
    else:
        train(args, env_config, drl_hyper_params)
        test(args, env_config, drl_hyper_params)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parsing the type of DRL/RL to be tested')
    parser.add_argument('-t', '--train', help='Train DRL/RL', required=True)
    parser.add_argument('-o', '--observe', help='Observe a trained DRL/RL')
    parser.add_argument('-f', '--folder', help='Logging folder')
    args = vars(parser.parse_args())
    main(args)