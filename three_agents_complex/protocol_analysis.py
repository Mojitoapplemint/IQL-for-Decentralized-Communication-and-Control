import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
import three_agents_complex_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from three_agents_complex_q import S_1, S_2, S_3, ACTIONS, FOLDER_NAME, get_action

successful_protocols = pd.read_csv('three_agents_complex/successful_protocols.csv')

A1_PROTOCOL={
    0:"Not Communicate",
    1:"Communicate to Agent 3",
    2:"Communicate to Agent 2",
    3:"Communicate to Both Agents"
}

A2_PROTOCOL={
    0:"Not Communicate",
    1:"Communicate to Agent 3",
    2:"Communicate to Agent 1",
    3:"Communicate to Both Agents"
}

A3_PROTOCOL={
    0:"Not Communicate",
    1:"Communicate to Agent 2",
    2:"Communicate to Agent 1",
    3:"Communicate to Both Agents"
}

protocol_dict = {}

for index, row in successful_protocols.iterrows():
    
    a1_protocol = {}

    a2_protocol = {}

    a3_protocol = {}

    for key in S_1:
        a1_protocol[key] = [0,0,0,0]
        
    for key in S_2:
        a2_protocol[key] = [0,0,0,0]

    for key in S_3:
        a3_protocol[key] = [0,0,0,0]
    
    # print(f"{index} / {len(successful_protocols)}", end="\r")
    protocol = row["Protocol"].replace("(","").replace(")","").replace(".0","").split(", ")
    protocol = [int(x) for x in protocol]
    
    q_1 = protocol[0:len(S_1)].copy()
    q_2 = protocol[len(S_1):len(S_1)+len(S_2)].copy()
    q_3 = protocol[len(S_1)+len(S_2):len(S_1)+len(S_2)+len(S_3)].copy()
    
    # print(len(q_1), len(q_2), len(q_3))

    env = gym.make('ThreeAgentsComplexEnv-v0', render_mode=None, string_mode="stats")
    
    for i in range(1000):
        terminated =False
        simulation_result = False
        
        v_state, info = env.reset()
        curr_event = info["curr_event"]
        string = info["word"]
        # print(string)
        
        _, agent_1_belief, agent_2_belief, agent_3_belief = v_state
        
        agent_1_in_dead_state, agent_2_in_dead_state, agent_3_in_dead_state = False, False, False
        
        communication_count = [0,0,0]
        
        while not terminated:
            if curr_event == 'a':
                agent_id = 1
                
                s_1 = S_1[(agent_1_belief, curr_event, agent_2_in_dead_state, agent_3_in_dead_state)]
                
                
                # Choosing action only based on the Q value; never explore
                a1_action = q_1[s_1]
                
                a1_protocol[(agent_1_belief, curr_event, agent_2_in_dead_state, agent_3_in_dead_state)][a1_action] += 1
                
                v_state, reward, terminated, simulation_result, info = env.step((agent_id, ACTIONS[a1_action]))

                
            if curr_event == 'b':
                agent_id = 2
                
                s_2 = S_2[(agent_2_belief, curr_event, agent_1_in_dead_state, agent_3_in_dead_state)]
                
                # Choosing action only based on the Q value; never explore
                a2_action = q_2[s_2]
                
                a2_protocol[(agent_2_belief, curr_event, agent_1_in_dead_state, agent_3_in_dead_state)][a2_action] += 1
                
                v_state, reward, terminated, simulation_result, info = env.step((agent_id, ACTIONS[a2_action]))

                
            if curr_event == 'c':
                agent_id = 3
                
                s_3 = S_3[(agent_3_belief, curr_event, agent_1_in_dead_state, agent_2_in_dead_state)]
                
                # Choosing action only based on the Q value; never explore
                a3_action = q_3[s_3]
                
                a3_protocol[(agent_3_belief, curr_event, agent_1_in_dead_state, agent_2_in_dead_state)][a3_action] += 1
                
                v_state, reward, terminated, simulation_result, info = env.step((agent_id, ACTIONS[a3_action]))

            
            if curr_event == 'x':
                agent_id = 13
                
                s_1 = S_1[(agent_1_belief, curr_event, agent_2_in_dead_state, agent_3_in_dead_state)]   
                s_3 = S_3[(agent_3_belief, curr_event, agent_1_in_dead_state, agent_2_in_dead_state)]
                
                if agent_2_in_dead_state:
                    a1_action = 0
                    a3_action = 0
                else:        
                    a1_action = q_1[s_1]
                    a3_action = q_3[s_3]
                
                a1_protocol[(agent_1_belief, curr_event, agent_2_in_dead_state, agent_3_in_dead_state)][a1_action] += 1
                a3_protocol[(agent_3_belief, curr_event, agent_1_in_dead_state, agent_2_in_dead_state)][a3_action] += 1
                
                joint_action = (a1_action, a3_action)
                
                v_state, reward, terminated, simulation_result, info = env.step((agent_id, joint_action))
                
                a1_protocol[(agent_1_belief, curr_event, agent_2_in_dead_state, agent_3_in_dead_state)][a1_action] += 1
                a3_protocol[(agent_3_belief, curr_event, agent_1_in_dead_state, agent_2_in_dead_state)][a3_action] += 1
            
            _, agent_1_belief, agent_2_belief, agent_3_belief = v_state
            agent_1_in_dead_state = agent_1_belief == -1
            agent_2_in_dead_state = agent_2_belief == -1
            agent_3_in_dead_state = agent_3_belief == -1
            curr_event = info["curr_event"]
    
    # print(q_2)
    # print(q_3)
    
    print("\n====================================================================") 
    # print(q_1)
    print("Agent 1 Protocol:")
    count=0
    for key in a1_protocol:
        total_count = sum(a1_protocol[key])
        if total_count > 0:
            if key[1] == 'a':
                print(f"{key}: {q_1[count]} -  {A1_PROTOCOL[np.argmax(a1_protocol[key])]}")
            elif key[1] == 'x':
                print(f"{key}: {q_1[count]} -  { 'Communicate' if np.argmax(a1_protocol[key]) else 'Not communicate'} to Agent 2")
    
    print("\nAgent 2 Protocol:")
    for key in a2_protocol:
        total_count = sum(a2_protocol[key])
        if total_count > 0:
            print(f"{key}: {A2_PROTOCOL[np.argmax(a2_protocol[key])]}")
    
    print("\nAgent 3 Protocol:")
    for key in a3_protocol:
        total_count = sum(a3_protocol[key])
        if total_count > 0:
            if key[1] == 'c':
                print(f"{key}: {A3_PROTOCOL[np.argmax(a3_protocol[key])]}")
            elif key[1] == 'x':
                print(f"{key}: { 'Communicate' if np.argmax(a3_protocol[key]) else 'Not communicate'} to Agent 2")


