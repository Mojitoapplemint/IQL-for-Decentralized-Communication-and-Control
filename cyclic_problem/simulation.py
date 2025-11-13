import numpy as np
import gymnasium as gym
import pandas as pd
import random
import cyclic_problem.cyclic_problem_env as cyclic_problem_env

q_1 = pd.read_csv("./complex_problem/demo_q1_table.csv")
q_2 = pd.read_csv("./complex_problem/demo_q2_table.csv")
q_1 = q_1.drop(q_1.columns[[0]], axis=1).to_numpy()
q_2 = q_2.drop(q_2.columns[[0]], axis=1).to_numpy()

ROW_NUMS = {
    (False, 0 ):0,
    (False, 1 ):1,
    (False, 2 ):2,
    (False, 3 ):3,
    (False, 4 ):4,
    (False, 5 ):5,
    (False,-1 ):6,
    (True,  0):7,
    (True,  1):8,
    (True,  2):9,
    (True,  3):10,
    (True,  4):11,
    (True,  5):12,
    (True, -1):13,
}

env = gym.make("ComplexEnv-v0", render_mode = "simulation", string_mode="simulation")

terminated = False
truncated = False

config, info = env.reset()

_, agent_1_observation, agent_2_observation = config

curr_symbol=info['input_alphabet']

agent_1_in_dead_state = False
agent_2_in_dead_state = False

while not(terminated or truncated):
    if curr_symbol == "a":
        
        agent_id=1
        agent_1_row_num = ROW_NUMS[(agent_2_in_dead_state, agent_1_observation)]
        
        if agent_2_in_dead_state:
            agent_1_communicate = 1
        else:
            agent_1_communicate = np.argmax(q_1[agent_1_row_num])
        
        config, reward, terminated, truncated, info = env.step((agent_id, agent_1_communicate))
        
        _, agent_1_observation, agent_2_observation = config
        
        agent_2_in_dead_state = agent_2_observation == -1
        
        curr_symbol=info['input_alphabet']
                    
    if curr_symbol == "b":
        agent_id=2
        agent_2_row_num = ROW_NUMS[(agent_1_in_dead_state, agent_2_observation)]
        
        if agent_1_in_dead_state:
            agent_2_communicate = 1
        else:        
            agent_2_communicate = np.argmax(q_2[agent_2_row_num])
        config, reward, terminated, truncated, info = env.step((agent_id, agent_2_communicate))
        
        _, agent_1_observation, agent_2_observation = config
        
        agent_1_in_dead_state = agent_1_observation == -1
        
        curr_symbol=info['input_alphabet']
        
