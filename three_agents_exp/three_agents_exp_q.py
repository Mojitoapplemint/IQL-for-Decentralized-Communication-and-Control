import numpy as np
import gymnasium as gym
import pandas as pd
import random
import three_agents_exp_env as three_agents_exp_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

FOLDER_NAME = 'three_agents_exp'

S_1 = {
    (1,'a', False, False):1,
    (3,'a', False, False):2,
    (4,'a', False, False):3,
    (6,'a', False, False):4,
    (7,'a', False, False):5,
    (8,'a', False, False):6,
    (9,'a', False, False):7,
    (10,'a', False, False):8,
    (11,'a', False, False):9,
    (12,'a', False, False):10,
    (13,'a', False, False):11,
    (14,'a', False, False):12,
    (15,'a', False, False):13,
    (16,'a', False, False):14,
    (17,'a', False, False):15,
    (18,'a', False, False):16,
    (19,'a', False, False):17,
    (-1,'a', False, False):18,
    (1,'a', True, False):19,
    (3,'a', True, False):20,
    (4,'a', True, False):21,
    (6,'a', True, False):22,
    (7,'a', True, False):23,
    (8,'a', True, False):24,
    (9,'a', True, False):25,
    (10,'a', True, False):26,
    (11,'a', True, False):27,
    (12,'a', True, False):28,
    (13,'a', True, False):29,
    (14,'a', True, False):30,
    (15,'a', True, False):31,
    (16,'a', True, False):32,
    (17,'a', True, False):33,
    (18,'a', True, False):34,
    (19,'a', True, False):35,
    (-1,'a', True, False):36,
    (1,'a', False, True):37,
    (3,'a', False, True):38,
    (4,'a', False, True):39,
    (6,'a', False, True):40,
    (7,'a', False, True):41,
    (8,'a', False, True):42,
    (9,'a', False, True):43,
    (10,'a', False, True):44,
    (11,'a', False, True):45,
    (12,'a', False, True):46,
    (13,'a', False, True):47,
    (14,'a',  False ,True ):48,
    (15,'a',  False ,True ):49,
    (16,'a',  False ,True ):50,
    (17,'a',  False ,True ):51,
    (18,'a',  False ,True ):52,
    (19,'a',  False ,True ):53,
    (-1,'a',  False ,True ):54,
    (1,'a',  True ,True ):55,
    (3,'a',  True ,True ):56,
    (4,'a',  True ,True ):57,
    (6,'a',  True ,True ):58,
    (7,'a',  True ,True ):59,
    (8,'a',  True ,True ):60,
    (9,'a',  True ,True ):61,
    (10,'a',  True ,True ):62,
    (11,'a',  True ,True ):63,
    (12,'a',  True ,True ):64,
    (13,'a',  True ,True ):65,
    (14,'a',  True ,True ):66,
    (15,'a',  True ,True ):67,
    (16,'a',  True ,True ):68,
    (17,'a',  True ,True ):69,
    (18,'a',  True ,True ):70,
    (19,'a',  True ,True ):71,
    (-1,'a',  True ,True ):72,
    
    (1,'x', False, False):73,
    (3,'x', False, False):74,
    (4,'x', False, False):75,
    (6,'x', False, False):76,
    (7,'x', False, False):77,
    (8,'x', False, False):78,
    (9,'x', False, False):79,
    (10,'x', False, False):80,
    (11,'x', False, False):81,
    (12,'x', False, False):82,
    (13,'x', False, False):83,
    (14,'x', False, False):84,
    (15,'x', False, False):85,
    (16,'x', False, False):86,
    (17,'x', False, False):87,
    (18,'x', False, False):88,
    (19,'x', False, False):89,
    (-1,'x', False, False):90,
    (1,'x', True, False):91,
    (3,'x', True, False):92,
    (4,'x', True, False):93,
    (6,'x', True, False):94,
    (7,'x', True, False):95,
    (8,'x', True, False):96,
    (9,'x', True, False):97,
    (10,'x', True, False):98,
    (11,'x', True, False):99,
    (12,'x', True, False):100,
    (13,'x', True, False):101,
    (14,'x', True, False):102,
    (15,'x', True, False):103,
    (16,'x', True, False):104,
    (17,'x', True, False):105,
    (18,'x', True, False):106,
    (19,'x', True, False):107,
    (-1,'x', True, False):108,
    (1,'x',  False ,True ):109,
    (3,'x',  False ,True ):110,
    (4,'x',  False ,True ):111,
    (6,'x',  False ,True ):112,
    (7,'x',  False ,True ):113,
    (8,'x',  False ,True ):114,
    (9,'x',  False ,True ):115,
    (10,'x',  False ,True ):116,
    (11,'x',  False ,True ):117,
    (12,'x',  False ,True ):118,
    (13,'x',  False ,True ):119,
    (14,'x',  False ,True ):120,
    (15,'x',  False ,True ):121,
    (16,'x',  False ,True ):122,
    (17,'x',  False ,True ):123,
    (18,'x',  False ,True ):124,
    (19,'x',  False ,True ):125,
    (-1,'x',  False ,True ):126,
    (1,'x',  True ,True ):127,
    (3,'x',  True ,True ):128,
    (4,'x',  True ,True ):129,
    (6,'x',  True ,True ):130,
    (7,'x',  True ,True ):131,
    (8,'x',  True ,True ):132,
    (9,'x',  True ,True ):133,
    (10,'x',  True ,True ):134,
    (11,'x',  True ,True ):135,
    (12,'x',  True ,True ):136,
    (13,'x',  True ,True ):137,
    (14,'x',  True ,True ):138,
    (15,'x',  True ,True ):139,
    (16,'x',  True ,True ):140,
    (17,'x',  True ,True ):141,
    (18,'x',  True ,True ):142,
    (19,'x',  True ,True ):143,
    (-1,'x',  True ,True ):144
}

S_3 = {
    (1,'c', False, False):1,
    (3,'c', False, False):2,
    (4,'c', False, False):3,
    (6,'c', False, False):4,
    (7,'c', False, False):5,
    (8,'c', False, False):6,
    (9,'c', False, False):7,
    (10,'c', False, False):8,
    (11,'c', False, False):9,
    (12,'c', False, False):10,
    (13,'c', False, False):11,
    (14,'c', False, False):12,
    (15,'c', False, False):13,
    (16,'c', False, False):14,
    (17,'c', False, False):15,
    (18,'c', False, False):16,
    (19,'c', False, False):17,
    (-1,'c', False, False):18,
    (1,'c', True, False):19,
    (3,'c', True, False):20,
    (4,'c', True, False):21,
    (6,'c', True, False):22,
    (7,'c', True, False):23,
    (8,'c', True, False):24,
    (9,'c', True, False):25,
    (10,'c', True, False):26,
    (11,'c', True, False):27,
    (12,'c', True, False):28,
    (13,'c', True, False):29,
    (14,'c', True, False):30,
    (15,'c', True, False):31,
    (16,'c', True, False):32,
    (17,'c', True, False):33,
    (18,'c', True, False):34,
    (19,'c', True, False):35,
    (-1,'c', True, False):36,
    (1,'c', False, True):37,
    (3,'c', False, True):38,
    (4,'c', False, True):39,
    (6,'c', False, True):40,
    (7,'c', False, True):41,
    (8,'c', False, True):42,
    (9,'c', False, True):43,
    (10,'c', False, True):44,
    (11,'c', False, True):45,
    (12,'c', False, True):46,
    (13,'c', False, True):47,
    (14,'c',  False ,True ):48,
    (15,'c',  False ,True ):49,
    (16,'c',  False ,True ):50,
    (17,'c',  False ,True ):51,
    (18,'c',  False ,True ):52,
    (19,'c',  False ,True ):53,
    (-1,'c',  False ,True ):54,
    (1,'c',  True ,True ):55,
    (3,'c',  True ,True ):56,
    (4,'c',  True ,True ):57,
    (6,'c',  True ,True ):58,
    (7,'c',  True ,True ):59,
    (8,'c',  True ,True ):60,
    (9,'c',  True ,True ):61,
    (10,'c',  True ,True ):62,
    (11,'c',  True ,True ):63,
    (12,'c',  True ,True ):64,
    (13,'c',  True ,True ):65,
    (14,'c',  True ,True ):66,
    (15,'c',  True ,True ):67,
    (16,'c',  True ,True ):68,
    (17,'c',  True ,True ):69,
    (18,'c',  True ,True ):70,
    (19,'c',  True ,True ):71,
    (-1,'c',  True ,True ):72,
    
    (1,'y', False, False):73,
    (3,'y', False, False):74,
    (4,'y', False, False):75,
    (6,'y', False, False):76,
    (7,'y', False, False):77,
    (8,'y', False, False):78,
    (9,'y', False, False):79,
    (10,'y', False, False):80,
    (11,'y', False, False):81,
    (12,'y', False, False):82,
    (13,'y', False, False):83,
    (14,'y', False, False):84,
    (15,'y', False, False):85,
    (16,'y', False, False):86,
    (17,'y', False, False):87,
    (18,'y', False, False):88,
    (19,'y', False, False):89,
    (-1,'y', False, False):90,
    (1,'y', True, False):91,
    (3,'y', True, False):92,
    (4,'y', True, False):93,
    (6,'y', True, False):94,
    (7,'y', True, False):95,
    (8,'y', True, False):96,
    (9,'y', True, False):97,
    (10,'y', True, False):98,
    (11,'y', True, False):99,
    (12,'y', True, False):100,
    (13,'y', True, False):101,
    (14,'y', True, False):102,
    (15,'y', True, False):103,
    (16,'y', True, False):104,
    (17,'y', True, False):105,
    (18,'y', True, False):106,
    (19,'y', True, False):107,
    (-1,'y', True, False):108,
    (1,'y',  False ,True ):109,
    (3,'y',  False ,True ):110,
    (4,'y',  False ,True ):111,
    (6,'y',  False ,True ):112,
    (7,'y',  False ,True ):113,
    (8,'y',  False ,True ):114,
    (9,'y',  False ,True ):115,
    (10,'y',  False ,True ):116,
    (11,'y',  False ,True ):117,
    (12,'y',  False ,True ):118,
    (13,'y',  False ,True ):119,
    (14,'y',  False ,True ):120,
    (15,'y',  False ,True ):121,
    (16,'y',  False ,True ):122,
    (17,'y',  False ,True ):123,
    (18,'y',  False ,True ):124,
    (19,'y',  False ,True ):125,
    (-1,'y',  False ,True ):126,
    (1,'y',  True ,True ):127,
    (3,'y',  True ,True ):128,
    (4,'y',  True ,True ):129,
    (6,'y',  True ,True ):130,
    (7,'y',  True ,True ):131,
    (8,'y',  True ,True ):132,
    (9,'y',  True ,True ):133,
    (10,'y',  True ,True ):134,
    (11,'y',  True ,True ):135,
    (12,'y',  True ,True ):136,
    (13,'y',  True ,True ):137,
    (14,'y',  True ,True ):138,
    (15,'y',  True ,True ):139,
    (16,'y',  True ,True ):140,
    (17,'y',  True ,True ):141,
    (18,'y',  True ,True ):142,
    (19,'y',  True ,True ):143,
    (-1,'y',  True ,True ):144
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

def get_action(q_table, agent_j_in_dead_state, agent_k_in_dead_state, row_num, epsilon):
    # Both agents are in dead state, only action [0,0] is possible
    if agent_j_in_dead_state and agent_k_in_dead_state:
        return 0  
    
    # If one agent is in dead state, limit actions for that agent 
    elif agent_j_in_dead_state:
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, 1)  # Explore
        else:
            return np.argmax(q_table[row_num][[0,1]])  # Exploit
    elif agent_k_in_dead_state:
        if random.uniform(0, 1) < epsilon:
            return  random.choice([0,2])  # Explore
        else:
            return  2*np.argmax(q_table[row_num][[0,2]])  # Exploit

    # Neither agent is in dead state, all actions possible
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # Explore
    else:
        return np.argmax(q_table[row_num])  # Exploit
    
def q_training(env, epochs=10000, alpha = 0.1, gamma=0.1, epsilon=0.1, print_process=False):
    q_1 = np.zeros((len(S_1), env.action_space.n))
    # q_2 = np.zeros((len(S_2), env.action_space.n))
    q_3 = np.zeros((len(S_3), env.action_space.n))
    
    for episode in range(epochs):
        if (print_process and episode%100==0):
            print(str(100*episode/epochs)+"%","done" , end="\r")
            
        config, info = env.reset()
        
        curr_event=info['curr_event']
        
        _, agent_1_belief, agent_2_belief, agent_3_belief = config
        
        prev_s_1, prev_s_2, prev_s_3 = -1, -1, -1
        
        terminated = False
        
        reward_1, reward_2, reward_3 = 0, 0, 0
        
        agent_1_in_dead_state, agent_2_in_dead_state, agent_3_in_dead_state = False, False, False
        
        a1_action, a2_action, a3_action = None, None, None

        
        while not terminated:
            if curr_event in A1_OBS:
                agent_id = 1
                
                s_1 = S_1[(agent_1_belief, curr_event, agent_2_in_dead_state, agent_3_in_dead_state)]
                
                if prev_s_1 != -1 :
                    # Q-value update for agent 1
                    q_1[prev_s_1][a1_action] += alpha * (reward_1 + gamma * np.max(q_1[s_1]) - q_1[prev_s_1][a1_action])
                    reward_1 = 0
                                
                a1_action = get_action(q_1, agent_j_in_dead_state=agent_2_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=s_1, epsilon=epsilon)

                
                config, reward, terminated, _, info = env.step((agent_id, ACTIONS[a1_action]))
                
                comm_cost, penalty = reward
                
                _, agent_1_belief, agent_2_belief, agent_3_belief = config
                
                reward_1 += comm_cost
                
                prev_s_1 = s_1
                
            # if curr_event in A2_OBS:
            #     agent_id = 2
                
            #     s_2 = S_2[(agent_2_belief, curr_event, agent_1_in_dead_state, agent_3_in_dead_state)]
                
            #     if prev_s_2 != -1 :
            #         # Q-value update for agent 2
            #         q_2[prev_s_2][a2_action] += alpha * (reward_2 + gamma * np.max(q_2[s_2]) - q_2[prev_s_2][a2_action])
            #         reward_2 = 0
                
            #     a2_action = get_action(q_2, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_3_in_dead_state, row_num=s_2, epsilon=epsilon)
                
            #     config, reward, terminated, _, info = env.step((agent_id, ACTIONS[a2_action]))
                
            #     comm_cost, penalty = reward
                
            #     _, agent_1_belief, agent_2_belief, agent_3_belief = config
                
            #     reward_2 += comm_cost
                
            #     prev_s_2 = s_2
                
            if curr_event in A3_OBS:
                agent_id = 3
               
                s_3 = S_3[(agent_3_belief, curr_event, agent_1_in_dead_state, agent_2_in_dead_state)]
                
                if prev_s_3 != -1 :
                    # Q-value update for agent 3
                    q_3[prev_s_3][a3_action] += alpha * (reward_3 + gamma * np.max(q_3[s_3]) - q_3[prev_s_3][a3_action])
                    reward_3 = 0
                
                a3_action = get_action(q_3, agent_j_in_dead_state=agent_1_in_dead_state, agent_k_in_dead_state=agent_2_in_dead_state, row_num=s_3, epsilon=epsilon)
                
                config, reward, terminated, _, info = env.step((agent_id, ACTIONS[a3_action]))
                
                comm_cost, penalty = reward
                
                _, agent_1_belief, agent_2_belief, agent_3_belief = config
                
                reward_3 += comm_cost
                
                prev_s_3 = s_3
         
            agent_1_in_dead_state = agent_1_belief == -1
            
            agent_2_in_dead_state = agent_2_belief == -1
            
            agent_3_in_dead_state = agent_3_belief == -1
            
            curr_event=info['curr_event']
    
        
        # Q-value update for agents who took action
        reward_1 += penalty
        q_1[prev_s_1][a1_action] += alpha * (reward_1 + gamma * 0 - q_1[prev_s_1][a1_action])

        # reward_2 += penalty
        # q_2[prev_s_2][a2_action] += alpha * (reward_2 + gamma * 0 - q_2[prev_s_2][a2_action])

        reward_3 += penalty
        q_3[prev_s_3][a3_action] += alpha * (reward_3 + gamma * 0 - q_3[prev_s_3][a3_action])
        


    # print(action_dict)
    return q_1, q_3






