import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from word_generator import RegexWordGenerator
import gymnasium as gym
import cyclic_problem_env

GAMMA = 0.1

m_bottom={
    1:"{1}",
    2:"{2}",
    3:"{1,3}",
    4:"{4}",
    5:"{5}",
    6:"{6,1}",
    7:"{7}",
    -1:"{-1}",
}

PHI = {
    (False, 1 ):0,
    (False, 2 ):1,
    (False, 3 ):2,
    (False, 4 ):3,
    (False, 5 ):4,
    (False, 6 ):5,
    (False, 7 ):6,
    (False,-1 ):7,
    (True, 1 ):8,
    (True, 2 ):9,
    (True, 3 ):10,
    (True, 4 ):11,
    (True, 5 ):12,
    (True, 6 ):13,
    (True, 7 ):14,
    (True,-1 ):15,
}



communication_counts_per_state = {
    1:[0,0,0,0],
    2:[0,0,0,0],
    3:[0,0,0,0],
    4:[0,0,0,0],
    5:[0,0,0,0],
    6:[0,0,0,0],
    7:[0,0,0,0],
    -1:[0,0,0,0],
}


q_1 = [1, 1, 1, 0, 0, 0, 0, 1]
q_2 = [1, 0, 0, 0, 1, 1, 0, 1]

env = gym.make("CyclicEnv2-v0", render_mode = None, string_mode="stats")

a1_communication_protocol = {
        1:[0,0,0,0],
        2:[0,0,0,0],
        3:[0,0,0,0],
        4:[0,0,0,0],
        5:[0,0,0,0],
        6:[0,0,0,0],
        7:[0,0,0,0],
        -1:[0,0,0,0],
    }

a2_communication_protocol = {
        1:[0,0],
        2:[0,0],
        3:[0,0],
        4:[0,0],
        5:[0,0],
        6:[0,0],
        7:[0,0],
        -1:[0,0],
    }

communicate_count = [0,0,0,0]

for i in range (1000):
    terminated = False
    simulation_result = False


    config, info = env.reset()

    global_state, agent_1_belief, agent_2_belief = config

    curr_symbol=info['input_alphabet']

    agent_1_prev_row_num = -1
    agent_2_prev_row_num = -1

    agent_1_in_dead_state = False
    agent_2_in_dead_state = False

    reward_1=0
    reward_2=0
    
    t_1=1
    t_2=1        
    
    while not (terminated):
        if curr_symbol=='a':
            
            agent_id=1

            if agent_2_in_dead_state:
                agent_1_communicate = 0
            else:
                agent_1_row_num = PHI[(agent_2_in_dead_state, agent_1_belief)]
                agent_1_communicate = q_1[agent_1_row_num]
                
            if agent_1_communicate ==1:
                communicate_count[0] += 1
                (a1_communication_protocol[agent_1_belief])[0] += 1
                communication_counts_per_state[agent_1_belief][0] += 1
            else:
                communicate_count[1] += 1
                (a1_communication_protocol[agent_1_belief])[1] += 1
                communication_counts_per_state[agent_1_belief][1] += 1


            config, reward, terminated, truncated, info = env.step((agent_id, agent_1_communicate))
            
            _, agent_1_belief, agent_2_belief = config
            
            agent_2_in_dead_state = agent_2_belief == -1
                
            reward_1 += reward
            
            curr_symbol=info['input_alphabet']
            
            agent_1_prev_row_num = agent_1_row_num
            
            
                    
        if curr_symbol=='b':

            agent_id=2
            
            if agent_1_in_dead_state:
                agent_2_communicate = 0
            else:
                agent_2_row_num = PHI[(agent_1_in_dead_state, agent_1_belief)]
                agent_2_communicate = q_2[agent_2_row_num]
            
            
            if agent_2_communicate ==1:
                communicate_count[2] += 1
                (a2_communication_protocol[agent_2_belief])[0] += 1  
                communication_counts_per_state[agent_2_belief][2] += 1
            else:
                communicate_count[3] += 1
                (a2_communication_protocol[agent_2_belief])[1] += 1  
                communication_counts_per_state[agent_2_belief][3] += 1
            
            
            config, reward, terminated, truncated, info = env.step((agent_id, agent_2_communicate))
            
            _, agent_1_belief, agent_2_belief = config
            
            agent_1_in_dead_state = agent_1_belief == -1
            
            reward_2 += reward
                            
            curr_symbol=info['input_alphabet']
            
            agent_2_prev_row_num = agent_2_row_num

communicate_count = pd.DataFrame(communicate_count, index=['Agent 1 Communicate Count', 'Agent 1 Not Communicate Count', 'Agent 2 Communicate Count', 'Agent 2 Not Communicate Count'])

print(communicate_count)


print("\nAgent 1 Communication Protocols:")
for belief_state in a1_communication_protocol:
    if (a1_communication_protocol[belief_state] != [0,0]):
        print("In state "+ m_bottom[belief_state]+ " Num Communicate: " + str(a1_communication_protocol[belief_state][0]) + " Num Not Communicate: " + str(a1_communication_protocol[belief_state][1]))
        
print("\nAgent 2 Communication Protocols:")

for belief_state in a2_communication_protocol:
    if (a2_communication_protocol[belief_state] != [0,0]):
        print("In state "+ m_bottom[belief_state]+ " Num Communicate: " + str(a2_communication_protocol[belief_state][0]) + " Num Not Communicate: " + str(a2_communication_protocol[belief_state][1]))

# for key, value in communication_counts_per_state.items():
#     print("State "+ m_bottom[key]+ " Agent 1: " + str(value[0]) + " / "+str(value[1]) + " Agent 2: " + str(value[2]) + " / "+str(value[3]))

    