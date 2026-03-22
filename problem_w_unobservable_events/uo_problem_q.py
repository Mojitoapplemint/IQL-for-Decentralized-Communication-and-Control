import numpy as np
import gymnasium as gym
import pandas as pd
import random
import  uo_problem_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

FOLDER_NAME = "problem_w_unobservable_events"

S_1={
    (1, 'a', False):0,
    (2, 'a', False):1,
    (3, 'a', False):2,
    (4, 'a', False):3,
    (5, 'a', False):4,
    (6, 'a', False):5,
    (7, 'a', False):6,
    (8, 'a', False):7,
    (9, 'a', False):8,
    (10,'a', False):9,
    (11,'a', False):10,
    (12,'a', False):11,
    (13,'a', False):12,
    (14,'a', False):13,
    (15,'a', False):14,
    (16,'a', False):15,
    (17,'a', False):16,
    (19,'a', False):17,
    (21,'a', False):18,
    (-1,'a', False):19,
    (1, 'c', False):20,
    (2, 'c', False):21,
    (3, 'c', False):22,
    (4, 'c', False):23,
    (5, 'c', False):24,
    (6, 'c', False):25,
    (7, 'c', False):26,
    (8, 'c', False):27,
    (9, 'c', False):28,
    (10,'c', False):29,
    (11,'c', False):30,
    (12,'c', False):31,
    (13,'c', False):32,
    (14,'c', False):33,
    (15,'c', False):34,
    (16,'c', False):35,
    (17,'c', False):36,
    (19,'c', False):37,
    (21,'c', False):38,
    (-1,'c', False):39,
    
    (1, 'a', True):40,
    (2, 'a', True):41,
    (3, 'a', True):42,
    (4, 'a', True):43,
    (5, 'a', True):44,
    (6, 'a', True):45,
    (7, 'a', True):46,
    (8, 'a', True):47,
    (9, 'a', True):48,
    (10,'a', True):49,
    (11,'a', True):50,
    (12,'a', True):51,
    (13,'a', True):52,
    (14,'a', True):53,
    (15,'a', True):54,
    (16,'a', True):55,
    (17,'a', True):56,
    (19,'a', True):57,
    (21,'a', True):58,
    (-1,'a', True):59,
    (1, 'c', True):60,
    (2, 'c', True):61,
    (3, 'c', True):62,
    (4, 'c', True):63,
    (5, 'c', True):64,
    (6, 'c', True):65,
    (7, 'c', True):66,
    (8, 'c', True):67,
    (9, 'c', True):68,
    (10,'c', True):69,
    (11,'c', True):70,
    (12,'c', True):71,
    (13,'c', True):72,
    (14,'c', True):73,
    (15,'c', True):74,
    (16,'c', True):75,
    (17,'c', True):76,
    (19,'c', True):77,
    (21,'c', True):78,
    (-1,'c', True):79,
}

S_2={
    (1, 'x',False):0,
    (2, 'x',False):1,
    (3, 'x',False):2,
    (4, 'x',False):3,
    (5, 'x',False):4,
    (6, 'x',False):5,
    (7, 'x',False):6,
    (8, 'x',False):7,
    (9, 'x',False):8,
    (10,'x',False):9,
    (11,'x',False):10,
    (12,'x',False):11,
    (13,'x',False):12,
    (14,'x',False):13,
    (15,'x',False):14,
    (16,'x',False):15,
    (17,'x',False):16,
    (19,'x',False):17,
    (21,'x',False):18,
    (-1,'x',False):19,
    (1, 'y',False):20,
    (2, 'y',False):21,
    (3, 'y',False):22,
    (4, 'y',False):23,
    (5, 'y',False):24,
    (6, 'y',False):25,
    (7, 'y',False):26,
    (8, 'y',False):27,
    (9, 'y',False):28,
    (10,'y',False):29,
    (11,'y',False):30,
    (12,'y',False):31,
    (13,'y',False):32,
    (14,'y',False):33,
    (15,'y',False):34,
    (16,'y',False):35,
    (17,'y',False):36,
    (19,'y',False):37,
    (21,'y',False):38,
    (-1,'y',False):39,
    (1, 'z',False):40,
    (2, 'z',False):41,
    (3, 'z',False):42,
    (4, 'z',False):43,
    (5, 'z',False):44,
    (6, 'z',False):45,
    (7, 'z',False):46,
    (8, 'z',False):47,
    (9, 'z',False):48,
    (10,'z',False):49,
    (11,'z',False):50,
    (12,'z',False):51,
    (13,'z',False):52,
    (14,'z',False):53,
    (15,'z',False):54,
    (16,'z',False):55,
    (17,'z',False):56,
    (19,'z',False):57,
    (21,'z',False):58,
    (-1,'z',False):59,
    (1, 's',False):60,
    (2, 's',False):61,
    (3, 's',False):62,
    (4, 's',False):63,
    (5, 's',False):64,
    (6, 's',False):65,
    (7, 's',False):66,
    (8, 's',False):67,
    (9, 's',False):68,
    (10,'s',False):69,
    (11,'s',False):70,
    (12,'s',False):71,
    (13,'s',False):72,
    (14,'s',False):73,
    (15,'s',False):74,
    (16,'s',False):75,
    (17,'s',False):76,
    (19,'s',False):77,
    (21,'s',False):78,
    (-1,'s',False):79,
    (1, 't',False):80,
    (2, 't',False):81,
    (3, 't',False):82,
    (4, 't',False):83,
    (5, 't',False):84,
    (6, 't',False):85,
    (7, 't',False):86,
    (8, 't',False):87,
    (9, 't',False):88,
    (10,'t',False):89,
    (11,'t',False):90,
    (12,'t',False):91,
    (13,'t',False):92,
    (14,'t',False):93,
    (15,'t',False):94,
    (16,'t',False):95,
    (17,'t',False):96,
    (19,'t',False):97,
    (21,'t',False):98,
    (-1,'t',False):99,
    (1, 'r',False):100,
    (2, 'r',False):101,
    (3, 'r',False):102,
    (4, 'r',False):103,
    (5, 'r',False):104,
    (6, 'r',False):105,
    (7, 'r',False):106,
    (8, 'r',False):107,
    (9, 'r',False):108,
    (10,'r',False):109,
    (11,'r',False):110,
    (12,'r',False):111,
    (13,'r',False):112,
    (14,'r',False):113,
    (15,'r',False):114,
    (16,'r',False):115,
    (17,'r',False):116,
    (19,'r',False):117,
    (21,'r',False):118,
    (-1,'r',False):119,
    
    (1, 'x',True):120,
    (2, 'x',True):121,
    (3, 'x',True):122,
    (4, 'x',True):123,
    (5, 'x',True):124,
    (6, 'x',True):125,
    (7, 'x',True):126,
    (8, 'x',True):127,
    (9, 'x',True):128,
    (10,'x',True):129,
    (11,'x',True):130,
    (12,'x',True):131,
    (13,'x',True):132,
    (14,'x',True):133,
    (15,'x',True):134,
    (16,'x',True):135,
    (17,'x',True):136,
    (19,'x',True):137,
    (21,'x',True):138,
    (-1,'x',True):139,
    (1, 'y',True):140,
    (2, 'y',True):141,
    (3, 'y',True):142,
    (4, 'y',True):143,
    (5, 'y',True):144,
    (6, 'y',True):145,
    (7, 'y',True):146,
    (8, 'y',True):147,
    (9, 'y',True):148,
    (10,'y',True):149,
    (11,'y',True):150,
    (12,'y',True):151,
    (13,'y',True):152,
    (14,'y',True):153,
    (15,'y',True):154,
    (16,'y',True):155,
    (17,'y',True):156,
    (19,'y',True):157,
    (21,'y',True):158,
    (-1,'y',True):159,
    (1, 'z',True):160,
    (2, 'z',True):161,
    (3, 'z',True):162,
    (4, 'z',True):163,
    (5, 'z',True):164,
    (6, 'z',True):165,
    (7, 'z',True):166,
    (8, 'z',True):167,
    (9, 'z',True):168,
    (10,'z',True):169,
    (11,'z',True):170,
    (12,'z',True):171,
    (13,'z',True):172,
    (14,'z',True):173,
    (15,'z',True):174,
    (16,'z',True):175,
    (17,'z',True):176,
    (19,'z',True):177,
    (21,'z',True):178,
    (-1,'z',True):179,
    (1, 's',True):180,
    (2, 's',True):181,
    (3, 's',True):182,
    (4, 's',True):183,
    (5, 's',True):184,
    (6, 's',True):185,
    (7, 's',True):186,
    (8, 's',True):187,
    (9, 's',True):188,
    (10,'s',True):189,
    (11,'s',True):190,
    (12,'s',True):191,
    (13,'s',True):192,
    (14,'s',True):193,
    (15,'s',True):194,
    (16,'s',True):195,
    (17,'s',True):196,
    (19,'s',True):197,
    (21,'s',True):198,
    (-1,'s',True):199,
    (1, 't',True):200,
    (2, 't',True):201,
    (3, 't',True):202,
    (4, 't',True):203,
    (5, 't',True):204,
    (6, 't',True):205,
    (7, 't',True):206,
    (8, 't',True):207,
    (9, 't',True):208,
    (10,'t',True):209,
    (11,'t',True):210,
    (12,'t',True):211,
    (13,'t',True):212,
    (14,'t',True):213,
    (15,'t',True):214,
    (16,'t',True):215,
    (17,'t',True):216,
    (19,'t',True):217,
    (21,'t',True):218,
    (-1,'t',True):219,
    (1, 'r',True):220,
    (2, 'r',True):221,
    (3, 'r',True):222,
    (4, 'r',True):223,
    (5, 'r',True):224,
    (6, 'r',True):225,
    (7, 'r',True):226,
    (8, 'r',True):227,
    (9, 'r',True):228,
    (10,'r',True):229,
    (11,'r',True):230,
    (12,'r',True):231,
    (13,'r',True):232,
    (14,'r',True):233,
    (15,'r',True):234,
    (16,'r',True):235,
    (17,'r',True):236,
    (19,'r',True):237,
    (21,'r',True):238,
    (-1,'r',True):239,
}

A1_OBS = ['a', 'c']

A2_OBS = ['x', 'y', 'z', 's', 't', 'r']

def get_action(q_table, opponent_in_dead_state, row_num, epsilon):
    if opponent_in_dead_state:
        return 0
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 1)  # Explore: random action
    else:
        return  np.argmax(q_table[row_num])  # Exploit: best action from Q-table
        

def q_training(env, epochs=10000, alpha=0.1, gamma=0.9, epsilon=0.1, print_process=False):
    q_1 = np.zeros((2*len(S_1), env.action_space.n))
    q_2 = np.zeros((2*len(S_2), env.action_space.n))  
    
    for epoch in range(epochs):
        
        if (print_process and epoch%100==0):
                print(str(100*epoch/epochs)+"%","done" , end="\r")
            
        config, info = env.reset()
        
        curr_event=info['curr_event']
        
        _, agent_1_belief, agent_2_belief = config
        
        prev_s_1 = -1
        prev_s_2 = -1
        
        terminated = False
        truncated = False
        
        agent_1_communicate = 0
        agent_2_communicate = 0
        
        reward_1 = 0
        reward_2 = 0
        
        agent_1_in_dead_state = False
        agent_2_in_dead_state = False
        
        while not (terminated or truncated):
            if curr_event in A1_OBS:
                
                agent_id=1
                s_1 = S_1[(agent_1_belief, curr_event, agent_2_in_dead_state)]

                
                if prev_s_1 != -1 :
                    # Q-value update for agent 1
                    q_1[prev_s_1][agent_1_communicate] += alpha * (reward_1 + gamma * np.max(q_1[s_1]) - q_1[prev_s_1][agent_1_communicate])
                    reward_1 = 0
                
                agent_1_communicate = get_action(q_1, agent_2_in_dead_state, s_1, epsilon)
                config, reward, terminated, truncated, info = env.step((agent_id, agent_1_communicate))
                
                _, agent_1_belief, agent_2_belief = config
                
                agent_2_in_dead_state = agent_2_belief == -1
                
                comm_cost, penalty = reward
                    
                reward_1 += comm_cost
                
                curr_event=info['curr_event']
                
                prev_s_1 = s_1
                            
            if curr_event in A2_OBS:
                agent_id=2
                s_2 = S_2[(agent_2_belief, curr_event, agent_1_in_dead_state)]

                
                if prev_s_2 != -1 :
                    # print(agent_2_prev_row_num, agent_2_communicate, reward_2)
                    
                    # Q-value update for agent 2
                    q_2[prev_s_2][agent_2_communicate] += alpha * (reward_2 + gamma * np.max(q_2[s_2]) - q_2[prev_s_2][agent_2_communicate])
                    reward_2 = 0
                
                agent_2_communicate = get_action(q_2, agent_1_in_dead_state, s_2, epsilon)
                # print(agent_2_row_num)
                config, reward, terminated, truncated, info = env.step((agent_id, agent_2_communicate))
                # print(reward)
                
                _, agent_1_belief, agent_2_belief = config
                
                agent_1_in_dead_state = agent_1_belief == -1
                
                comm_cost, penalty = reward
                
                reward_2 += comm_cost
                                
                curr_event=info['curr_event']
                
                prev_s_2 = s_2
        
        reward_1 += penalty
        reward_2 += penalty
        
        
        # Final Q-value updates
        q_1[prev_s_1][agent_1_communicate] += alpha * (reward_1 + gamma * 0 - q_1[prev_s_1][agent_1_communicate])
        q_2[prev_s_2][agent_2_communicate] += alpha * (reward_2 + gamma * 0 - q_2[prev_s_2][agent_2_communicate])

    return q_1, q_2

