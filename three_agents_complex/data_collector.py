import numpy as np
import gymnasium as gym
import pandas as pd
import three_agents_complex_env
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from three_agents_complex_q import q_training, get_action, S_1, S_2, S_3, ACTIONS

success_dict = {}
result_dict = {}
session_count = 1000
for i in range(session_count):
    print(str(100*i/session_count)+"%","done" , end="\r")
    env = gym.make('ThreeAgentsComplexEnv-v0', render_mode=None, string_mode="training")
    
    q_1, q_2, q_3 = q_training(env, epochs=100000, alpha=0.001, gamma=0.9, epsilon=0.1)
    
    env = gym.make('ThreeAgentsComplexEnv-v0', render_mode=None, string_mode="simulation")
    
    fail_count = 0
    
    test_session = 100
    
    for _ in range(test_session):
        terminated = False
        simulation_result = False

        state, info = env.reset()
        
        curr_event = info["curr_event"]
        
        _, agent_1_belief, agent_2_belief, agent_3_belief = state

        agent_1_in_dead_state, agent_2_in_dead_state, agent_3_in_dead_state = False, False, False
        
        # while not terminated:
            