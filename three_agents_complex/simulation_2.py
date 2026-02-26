import numpy as np
import gymnasium as gym
import pandas as pd
import random
import three_agents_complex_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


env = gym.make('ThreeAgentsComplexEnv-v0', render_mode=None, string_mode="stats")

fail_count = 0

for i in range(1000):
    state, info = env.reset()

    curr_event = info["curr_event"]
    # word = info["word"]

    _, agent_1_belief, agent_2_belief, agent_3_belief = state

    agent_1_in_dead_state, agent_2_in_dead_state, agent_3_in_dead_state = False, False, False

    terminated = False
    simulation_result = False

    count = [0,0,0]

    a1_return, a2_return, a3_return = 0, 0, 0

    while not terminated:
        if curr_event == 'a':
            agent_id = 1
            
            a1_action = [1,0]
            
            count[0] +=np.sum(a1_action)
            
            config, reward, terminated, simulation_result, info = env.step((agent_id, a1_action))
            
            comm_cost, penalty = reward
            
            a1_return += comm_cost
            
            curr_event=info['curr_event']


        if curr_event == 'b':
            agent_id = 2
            
            a2_action = [0,0]
            
            count[1] +=np.sum(a2_action)
            
            config, reward, terminated, simulation_result, info = env.step((agent_id, a2_action))
            
            comm_cost, penalty = reward
            
            a2_return += comm_cost
            
            curr_event=info['curr_event']
            
        if curr_event == 'c':
            agent_id = 3
            
            a3_action = [0,1]
            
            count[2] +=np.sum(a3_action)
            
            config, reward, terminated, simulation_result, info = env.step((agent_id, a3_action))
            
            comm_cost, penalty = reward
            
            a3_return += comm_cost
            
            curr_event=info['curr_event']
        
        if curr_event == 'x':
            agent_id = 13
            
            a1_action = 1
            a3_action = 0        
            
            count[0] += a1_action
            count[2] += a3_action
            
            joint_action = (a1_action, a3_action)
            
            config, reward, terminated, simulation_result, info = env.step((agent_id, joint_action))
            
            comm_cost, penalty = reward
            
            comm_cost_1 , comm_cost_3 = comm_cost
            
            a1_return += comm_cost_1
            a3_return += comm_cost_3
        
        _, agent_1_belief, agent_2_belief, agent_3_belief = config
        
        agent_1_in_dead_state = agent_1_belief == -1
        
        agent_2_in_dead_state = agent_2_belief == -1
        
        agent_3_in_dead_state = agent_3_belief == -1
        
        curr_event=info['curr_event']

    if not simulation_result:
        fail_count += 1
        
        # print(a1_return, a2_return, a3_return)
        # print(count)

    a1_return += penalty

    a2_return += penalty

    a3_return += penalty


# print()
print(count)
print(a1_return, a2_return, a3_return)
print(f"Fail count: {fail_count}/1000")