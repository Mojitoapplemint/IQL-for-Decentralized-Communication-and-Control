import numpy as np
import gymnasium as gym
import pandas as pd
import three_agents_ls_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from three_agents_ls_q import S_1, S_3, ACTIONS,A1_OBS, A3_OBS, get_action, q_training, FOLDER_NAME


successful_protocol_dict = {}
failed_protocol_dict = {}
result_dict = {}
session_count = 10
for i in range(session_count):
    print(f"{i}/{session_count} done", end="\r")
    env = gym.make('ThreeAgentsLSEnv-v1', render_mode=None, string_mode="training")
    
    # q_1, q_3 = q_training(env, epochs=50000, alpha=0.001, gamma=0.5, min_epsilon=0.01, print_process=False)
    q_1, q_3 = q_training(env, epochs=250000, alpha=0.0005, gamma=0.5, min_epsilon=0.01, print_process=False)

    
    env = gym.make('ThreeAgentsLSEnv-v1', render_mode=None, string_mode="simulation")
    
    fail_count = 0
    
    test_session = 4
    
    for _ in range(test_session):
        # print("here")
        terminated = False
        simulation_result = False

        v_state, info = env.reset()
        
        curr_event = info["curr_event"]
        
        _, agent_1_belief, agent_2_belief, agent_3_belief = v_state
        
        while not terminated:
            if curr_event in A1_OBS:
                agent_id = 1        
                
                s_1 = S_1[(agent_1_belief,curr_event)]
                
                # Choosing action only based on the Q value; never explore
                a1_action = get_action(q_1,state=s_1, epsilon=0)
                
                a1_action = ACTIONS[a1_action]
                                
                v_state, reward, terminated, simulation_result, info = env.step((agent_id, a1_action))
                
            if curr_event in A3_OBS:
                agent_id = 3
                
                s_3 = S_3[(agent_3_belief, curr_event)]
                
                # Choosing action only based on the Q value; never explore
                a3_action = get_action(q_3, state=s_3, epsilon=0)
                
                a3_action = ACTIONS[a3_action]
                
                v_state, reward, terminated, simulation_result, info = env.step((agent_id, a3_action))
                
            
            _, agent_1_belief, agent_2_belief, agent_3_belief = v_state
        
            curr_event=info['curr_event']
    
    
        if not simulation_result:
            fail_count += 1
    
        if result_dict.get(tuple(v_state)) is None:
            result_dict[tuple(v_state)] = 1
        else:
            result_dict[tuple(v_state)] += 1



    a1_protocol = np.zeros((q_1.shape[0]), dtype=int)
    a3_protocol = np.zeros((q_3.shape[0]), dtype=int)
    
    for i in range(q_1.shape[0]):
        if list(q_1[i]).count(0)!=4:
            a1_protocol[i]=np.argmax([item for item in q_1[i] if item !=0])
        else:
            a1_protocol[i]=0
    
    for i in range(q_3.shape[0]):
        if list(q_3[i]).count(0)!=4:
            a3_protocol[i]=np.argmax([item for item in q_3[i] if item !=0])
        else:
            a3_protocol[i]=0
    
    protocol_key = (tuple(a1_protocol), tuple(a3_protocol))
    
    if fail_count == 0:
        if successful_protocol_dict.get(protocol_key) is None:
            successful_protocol_dict[protocol_key] = 1
        else:
            successful_protocol_dict[protocol_key] += 1
    else:
        if failed_protocol_dict.get(protocol_key) is None:
            failed_protocol_dict[protocol_key] = 1
        else:
            failed_protocol_dict[protocol_key] += 1

successful_protocols_df = pd.DataFrame(list(successful_protocol_dict.items()), columns=['Protocol', 'Counts'])

failed_protocol_df = pd.DataFrame(list(failed_protocol_dict.items()), columns=['Protocol', 'Counts'])

# print(successful_protocol_dict)

print(f"{np.sum(successful_protocols_df['Counts'])}/{session_count} sessions converged to a successful protocol.")

for key in result_dict:
    print(f"<{key[0]}, {key[1]}, {key[2]}, {key[3]}> => Count: {result_dict[key]}")

successful_protocols_df.to_csv(f'{FOLDER_NAME}/successful_protocols_exp_3.csv', index=False)
failed_protocol_df.to_csv(f'{FOLDER_NAME}/failed_protocols_exp_3.csv', index=False)