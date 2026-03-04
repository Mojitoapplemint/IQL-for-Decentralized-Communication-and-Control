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


# regexgen = RegexWordGenerator(max_star=5)

# string_list = []

# for i in range(1000):
#     string_list.append(regexgen.generate_simulation_word())
    
# df = pd.DataFrame(string_list, columns=["strings"])
# df.to_csv("second_cyclic_problem/strings.csv", index=False)


successful_protocols = pd.read_csv("cyclic_problem_w_unobservable_events/simulation_successful_protocols.csv")
# successful_protocols = pd.read_csv("cyclic_problem_w_unobservable_events/baselines.csv")


# print(successful_protocols)

success_return_values_x = []
success_return_values_y = []
joint_return_values = []

communicate_counts = []

a1_protocol_list=[]
a2_protocol_list=[]

T_state_dict_list = []

communication_counts_per_state = {
     1 :[0,0,0,0],
     2 :[0,0,0,0],
     3 :[0,0,0,0],
     4 :[0,0,0,0],
     5 :[0,0,0,0],
     6 :[0,0,0,0],
     7 :[0,0,0,0],
    -1:[0,0,0,0],
}

for index, row in successful_protocols.iterrows():
    # print(f"{index} / {len(successful_protocols)}")

    
    protocol = row["Communication Protocols"].replace("(","").replace(")","").split(", ")
    protocol = [int(x) for x in protocol]
    
    # print(protocol)
    
    q_1 = protocol[:16]
    q_2 = protocol[16:]
    
    cumulative_reward = [0,0]
    
    env = gym.make("CyclicEnv-v0", render_mode = None, string_mode="stats")
    
    a1_communication_protocol = {
           ( 1, False):[0,0],
           ( 2, False):[0,0],
           ( 3, False):[0,0],
           ( 4, False):[0,0],
           ( 5, False):[0,0],
           ( 6, False):[0,0],
           ( 7, False):[0,0],
           (-1, False):[0,0],
           ( 1, True):[0,0],
           ( 2, True):[0,0],
           ( 3, True):[0,0],
           ( 4, True):[0,0],
           ( 5, True):[0,0],
           ( 7, True):[0,0],
           ( 6, True):[0,0],
           (-1, True):[0,0],
        }
    
    a2_communication_protocol = {
            ( 1, False):[0,0],
            ( 2, False):[0,0],
            ( 3, False):[0,0],
            ( 4, False):[0,0],
            ( 5, False):[0,0],
            ( 6, False):[0,0],
            ( 7, False):[0,0],
            (-1, False):[0,0],
            ( 1, True):[0,0],
            ( 2, True):[0,0],
            ( 3, True):[0,0],
            ( 4, True):[0,0],
            ( 5, True):[0,0],
            ( 6, True):[0,0],
            ( 7, True):[0,0],
            (-1, True):[0,0],
        }
    
    T_state_dict={
        "(3, 3, 3)":0,
        "(7, 7, 7)":0,
        "(4, -1, -1)":0,
        "(7, -1, -1)":0,
        "(3, 3, -1)":0,
        "(3, 3, 5)":0,
        "(7, 7, -1)":0,
        "(3, -1, 3)":0,
        "(7, -1, 7)":0,
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
                
                if agent_1_prev_row_num != -1 :
                    # cumulative_reward[0] += (GAMMA**t_1)*reward_1
                    cumulative_reward[0] += reward_1
                    t_1+=1
                    reward_1 = 0

                if agent_2_in_dead_state:
                    agent_1_communicate = 0
                else:
                    agent_1_row_num = PHI[(agent_2_in_dead_state, agent_1_belief)]
                    agent_1_communicate = q_1[agent_1_row_num]
                    
                if agent_1_communicate ==1:
                    communicate_count[0] += 1
                    (a1_communication_protocol[agent_1_belief, agent_2_in_dead_state])[0] += 1
                    communication_counts_per_state[agent_1_belief][0] += 1
                else:
                    communicate_count[1] += 1
                    (a1_communication_protocol[agent_1_belief, agent_2_in_dead_state])[1] += 1
                    communication_counts_per_state[agent_1_belief][1] += 1


                config, reward, terminated, truncated, info = env.step((agent_id, agent_1_communicate))
                
                system_state, agent_1_belief, agent_2_belief = config
                
                agent_2_in_dead_state = agent_2_belief == -1
                    
                reward_1 += reward
                
                curr_symbol=info['input_alphabet']
                
                agent_1_prev_row_num = agent_1_row_num
                
                
                       
            if curr_symbol=='b':

                agent_id=2
                
                if agent_2_prev_row_num != -1 :
                    # cumulative_reward[1] += (GAMMA**t_2)*reward_2
                    cumulative_reward[1] += reward_2
                    # return_value[1] += reward_2
                    t_2+=1
                    reward_2 = 0
                
                if agent_1_in_dead_state:
                    agent_2_communicate = 0
                else:
                    agent_2_row_num = PHI[(agent_1_in_dead_state, agent_2_belief)]
                    agent_2_communicate = q_2[agent_2_row_num]
                
                
                if agent_2_communicate ==1:
                    
                    communicate_count[2] += 1
                    (a2_communication_protocol[agent_2_belief, agent_1_in_dead_state])[0] += 1  
                    communication_counts_per_state[agent_2_belief][2] += 1
                else:
                    communicate_count[3] += 1
                    (a2_communication_protocol[agent_2_belief, agent_1_in_dead_state])[1] += 1  
                    communication_counts_per_state[agent_2_belief][3] += 1
                
                
                config, reward, terminated, truncated, info = env.step((agent_id, agent_2_communicate))
                
                system_state, agent_1_belief, agent_2_belief = config
                
                agent_1_in_dead_state = agent_1_belief == -1
                
                reward_2 += reward
                                
                curr_symbol=info['input_alphabet']
                
                agent_2_prev_row_num = agent_2_row_num

        
        if agent_id ==1:
            reward_2 += reward
        else:
            reward_1 += reward
        
        # cumulative_reward[0] += (GAMMA**t_1)*reward_1
        # cumulative_reward[1] += (GAMMA**t_2)*reward_2
        
        cumulative_reward[0] += reward_1
        cumulative_reward[1] += reward_2
        
        final_state = str(tuple((system_state, agent_1_belief, agent_2_belief)))

        T_state_dict[final_state] += 1     
    
    T_state_dict_list.append(T_state_dict)
    
    # print(dead_state_enter_count)
    communicate_counts.append(1/1000*np.array(communicate_count))
    
    a1_protocol_list.append(a1_communication_protocol)
    a2_protocol_list.append(a2_communication_protocol)

    
    cumulative_reward[0] = np.round(cumulative_reward[0]/1000, 2)
    cumulative_reward[1] = np.round(cumulative_reward[1]/1000, 2)
    success_return_values_x.append(cumulative_reward[0])
    success_return_values_y.append(cumulative_reward[1])
    joint_return_values.append((cumulative_reward[0], cumulative_reward[1]))
    
# print(communicate_counts)

successful_protocols["Agent 1 Average Cumulative Reward"] = success_return_values_x
successful_protocols["Agent 2 Average Cumulative Reward"] = success_return_values_y

print(pd.DataFrame(joint_return_values, columns=['Agent 1 Return', 'Agent 2 Return']).value_counts())
 
plt.figure(figsize=(10,6))
plt.scatter(success_return_values_x[2:], success_return_values_y[2:], color='blue', label='Successful Protocols')
plt.scatter(success_return_values_x[0], success_return_values_y[0], color='red', label='Full Communication')
plt.scatter(success_return_values_x[1], success_return_values_y[1], color='green', label='No communication')

for i in range(len(success_return_values_x)):
    plt.text(success_return_values_x[i], success_return_values_y[i], f"{(success_return_values_x[i], success_return_values_y[i])}", fontsize=6, ha='left',va='bottom', color='purple', rotation=45)

plt.xlabel('Agent 1 Average Cumulative Reward per word')
plt.ylabel('Agent 2 Average Cumulative Reward per word')
# plt.title(' of Communication Protocols')
plt.legend()
plt.grid(True)
plt.savefig('cyclic_problem_w_unobservable_events/cumulative_reward_for_successful_protocols.png')
plt.show()

communicate_counts = pd.DataFrame(communicate_counts, columns=['Agent 1 Communicate Count', 'Agent 1 Not Communicate Count', 'Agent 2 Communicate Count', 'Agent 2 Not Communicate Count'])

print(communicate_counts)

# print(T_state_dict_list)

T_state_df = pd.DataFrame(T_state_dict_list, columns=["(3, 3, 3)","(7, 7, 7)","(4, -1, -1)","(7, -1, -1)","(3, 3, -1)", "(3, 3, 5)", "(7, 7, -1)","(3, -1, 3)","(7, -1, 7)"])

print(T_state_df)

successful_protocols = pd.concat([successful_protocols, communicate_counts, T_state_df], axis=1)

successful_protocols.to_csv("cyclic_problem_w_unobservable_events/simulation_successful_protocols_with_stats.csv", index=False)

for i in range(len(a1_protocol_list)):
    a1_protocol = a1_protocol_list[i]
    a2_protocol = a2_protocol_list[i]
    
    
    print(f"\n================== {i}'th protocol ==================\nAgent 1 Communication Protocol:")
    for belief_state in a1_protocol:
        # print(belief_state)
        if (a1_protocol[belief_state] != [0,0] and not belief_state[1]):
            print("In state ("+ m_bottom[belief_state[0]]+ ", "+str(belief_state[1])+ ") Num Communicate: " + str(a1_protocol[belief_state][0]) + " Num Not Communicate: " + str(a1_protocol[belief_state][1]))

    print(f"\nAgent 2 Communication Protocol:")
    for belief_state in a2_protocol:
        if (a2_protocol[belief_state] != [0,0] and not belief_state[1]):
            print("In state ("+ m_bottom[belief_state[0]]+ ", "+str(belief_state[1])+ ") Num Communicate: " + str(a2_protocol[belief_state][0]) + " Num Not Communicate: " + str(a2_protocol[belief_state][1]))


# print("Agent 1 Communication Protocols:")
# for protocol in a1_protocol_list:
#     if a1_protocol_list.index(protocol)not in [0,2,3,4,5,10]:
#         continue
#     # print(f"\n"+str(protocol))
#     print()
#     for belief_state in protocol:
#         # print(belief_state)
#         if (protocol[belief_state] != [0,0]):
#             print("In state ("+ m_bottom[belief_state[0]]+ ", "+str(belief_state[1])+ ") Num Communicate: " + str(protocol[belief_state][0]) + " Num Not Communicate: " + str(protocol[belief_state][1]))
        
# print("\nAgent 2 Communication Protocols:")
# for protocol in a2_protocol_list:
#     if a2_protocol_list.index(protocol)not in [0,2,3,4,5,10]:
#         continue
#     # print(f"\n"+str(protocol))
#     print()
#     for belief_state in protocol:
#         if (protocol[belief_state] != [0,0]):
#             print("In state ("+ m_bottom[belief_state[0]]+ ", "+str(belief_state[1])+ ") Num Communicate: " + str(protocol[belief_state][0]) + " Num Not Communicate: " + str(protocol[belief_state][1]))


# for key, value in communication_counts_per_state.items():
#     print("State "+ m_bottom[key]+ " Agent 1: " + str(value[0]) + " / "+str(value[1]) + " Agent 2: " + str(value[2]) + " / "+str(value[3]))
