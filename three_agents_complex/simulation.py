import numpy as np
import gymnasium as gym
import pandas as pd
import random
import three_agents_complex_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from three_agents_complex_q import S_1, S_2, S_3, ACTIONS, get_action

q_1 = pd.read_csv('three_agents_complex/three_agents_complex_q1.csv').to_numpy()
q_2 = pd.read_csv('three_agents_complex/three_agents_complex_q2.csv').to_numpy()
q_3 = pd.read_csv('three_agents_complex/three_agents_complex_q3.csv').to_numpy()


env = gym.make('ThreeAgentsComplexEnv-v0', render_mode="human", string_mode="simulation")

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
        
        # Choosing action only based on the Q value; never explore
        a1_action = get_action(q_1, agent_j_in_dead_state=agent_2_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=s_1, epsilon=0)
        
        a1_action = ACTIONS[a1_action]
        
        count[0] +=np.sum(a1_action)
        
        config, reward, terminated, _, info = env.step((agent_id, a1_action))
        
        comm_cost, penalty = reward
        
        a1_return += comm_cost
        
    if curr_event == 'b':
        agent_id = 2
        
        # Choosing action only based on the Q value; never explore
        a2_action = get_action(q_2, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=s_2, epsilon=0)
        
        a2_action = ACTIONS[a2_action]
        count[1] +=np.sum(a2_action)
        config, reward, terminated, _, info = env.step((agent_id, a2_action))
        
        comm_cost, penalty = reward
                
        a2_return += comm_cost
        
    if curr_event == 'c':
        agent_id = 3
        
        # Choosing action only based on the Q value; never explore
        a3_action = get_action(q_3, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_2_in_dead_state, row_num=s_3, epsilon=0)
        
        a3_action = ACTIONS[a3_action]
        count[2] +=np.sum(a3_action)
        
        config, reward, terminated, _, info = env.step((agent_id, a3_action))
        
        comm_cost, penalty = reward
        
        a3_return += comm_cost
    
    if curr_event == 'x':
        agent_id = 13
        
        if agent_2_in_dead_state:
            a1_action = 0
            a3_action = 0
        else:
            a1_action = np.argmax(q_1[s_1][[0,1]])            
            a3_action = np.argmax(q_3[s_3][[0,1]])            
        
        count[0] += a1_action
        count[2] += a3_action
        
        joint_action = (a1_action, a3_action)
        
        config, reward, terminated, _, info = env.step((agent_id, joint_action))
        
        comm_cost, penalty = reward
        
        comm_cost_1 , comm_cost_3 = comm_cost
        
        a1_return += comm_cost_1
        a3_return += comm_cost_3
    
    _, agent_1_belief, agent_2_belief, agent_3_belief = config
    
    agent_1_in_dead_state = agent_1_belief == -1
    
    agent_2_in_dead_state = agent_2_belief == -1
    
    agent_3_in_dead_state = agent_3_belief == -1
    
    curr_event=info['curr_event']
    
    # print(a1_return, a2_return, a3_return)
    # print(count)

a1_return += penalty

a2_return += penalty

a3_return += penalty


# print()
print(count)
print(a1_return, a2_return, a3_return)