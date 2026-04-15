import numpy as np
import gymnasium as gym
import pandas as pd
import random
import three_agents_ls_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


FOLDER_NAME = 'three_agents_long_short'

S_1 = {
    (1, 'a'):1,
    (2, 'a'):2,
    (3, 'a'):3,
    (4, 'a'):4,
    (5, 'a'):5,
    (6, 'a'):6,
    (7, 'a'):7,
    (8, 'a'):8,
    (9, 'a'):9,
    (10,'a'):10,
    (11,'a'):11,
    (12,'a'):12,
    (13,'a'):13,
    (-1,'a'):14,
    (1, 'x'):15,
    (2, 'x'):16,
    (3, 'x'):17,
    (4, 'x'):18,
    (5, 'x'):19,
    (6, 'x'):20,
    (7, 'x'):21,
    (8, 'x'):22,
    (9, 'x'):23,
    (10,'x'):24,
    (11,'x'):25,
    (12,'x'):26,
    (13,'x'):27,
    (-1,'x'):28,
}

S_3 = {
    (1, 'c'):1,
    (2, 'c'):2,
    (3, 'c'):3,
    (4, 'c'):4,
    (5, 'c'):5,
    (6, 'c'):6,
    (7, 'c'):7,
    (8, 'c'):8,
    (9, 'c'):9,
    (10,'c'):10,
    (11,'c'):11,
    (12,'c'):12,
    (13,'c'):13,
    (-1,'c'):14,
    (1, 'y'):15,
    (2, 'y'):16,
    (3, 'y'):17,
    (4, 'y'):18,
    (5, 'y'):19,
    (6, 'y'):20,
    (7, 'y'):21,
    (8, 'y'):22,
    (9, 'y'):23,
    (10,'y'):24,
    (11,'y'):25,
    (12,'y'):26,
    (13,'y'):27,
    (-1,'y'):28,
}


A1_OBS = ['a', 'x']
A2_OBS = ['s']
A3_OBS = ['c', 'y']

ACTIONS = {
    0:[0,0],
    1:[0,1],
    2:[1,0],
    3:[1,1],
}

ACTIONS_INV ={
    (0,0):0,
    (0,1):1,
    (1,0):2,
    (1,1):3,
}



def epsilon_decay(min_epsilon, episode, max_epochs):
    if episode <= 0.3*max_epochs:
        return 1.0
    
    initial_epsilon = 1.0
    return max(min_epsilon, initial_epsilon-(episode/(max_epochs)))
    # return min_epsilon

def get_action(q_table, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # Explore: random action
    else:
        return  np.argmax(q_table[state])  # Exploit: best action from Q-table


def q_training(env, epochs=10000, alpha = 0.1, gamma=0.1, min_epsilon=0.1, print_process=False):
    q_1 = np.zeros((len(S_1), env.action_space.n))
    # q_2 = np.zeros((len(S_2), env.action_space.n))
    q_3 = np.zeros((len(S_3), env.action_space.n))
    
    for episode in range(epochs):
        if (print_process and episode%100==0):
            print(str(100*episode/epochs)+"%","done" , end="\r")
        
        config, info = env.reset()
        
        curr_event=info['curr_event']
        
        _, agent_1_belief, agent_2_belief, agent_3_belief = config
        
        s_1, s_3 = -1, -1
        
        terminated = False
        
        reward_1, reward_3 = 0, 0
        
        a1_action, a3_action = None, None

        epsilon = epsilon_decay(min_epsilon, episode, epochs)
        # if (episode%10000==0):
        #     print(f"Episode: {episode}, Epsilon: {epsilon}")
        
        while not terminated:
            if curr_event in A1_OBS:
                agent_id = 1
                
                next_s_1 = S_1[(agent_1_belief, curr_event)]
                
                if s_1 != -1 :
                    # Q-value update for agent 1
                    q_1[s_1][a1_action] += alpha * (reward_1 + gamma * np.max(q_1[next_s_1]) - q_1[s_1][a1_action])

                    reward_1 = 0
                                
                a1_action = get_action(q_1, state=next_s_1, epsilon=epsilon)
                
                config, reward, terminated, _, info = env.step((agent_id, ACTIONS[a1_action]))
                
                comm_cost, penalty = reward
                
                _, agent_1_belief, agent_2_belief, agent_3_belief = config
                
                reward_1 += comm_cost
                
                s_1 = next_s_1
                
            if curr_event in A3_OBS:
                agent_id = 3
               
                next_s_3 = S_3[(agent_3_belief, curr_event)]
                
                if s_3 != -1 :
                    # Q-value update for agent 3
                    q_3[s_3][a3_action] += alpha * (reward_3 + gamma * np.max(q_3[next_s_3]) - q_3[s_3][a3_action])
                    reward_3 = 0
                
                a3_action = get_action(q_3, state=next_s_3, epsilon=epsilon)
                
                config, reward, terminated, _, info = env.step((agent_id, ACTIONS[a3_action]))
                
                comm_cost, penalty = reward
                
                _, agent_1_belief, agent_2_belief, agent_3_belief = config
                
                reward_3 += comm_cost
                
                s_3 = next_s_3
            
            curr_event=info['curr_event']
    
        
        # Q-value update for agents who took action
        reward_1 += penalty
        q_1[s_1][a1_action] += alpha * (reward_1 + gamma * 0 - q_1[s_1][a1_action])

        # reward_2 += penalty
        # q_2[prev_s_2][a2_action] += alpha * (reward_2 + gamma * 0 - q_2[prev_s_2][a2_action])
        reward_3 += penalty
        q_3[s_3][a3_action] += alpha * (reward_3 + gamma * 0 - q_3[s_3][a3_action])
        
        
    # print(a1_action_count)
    # print(a3_action_count)
    # print(action_dict)
    return q_1, q_3






