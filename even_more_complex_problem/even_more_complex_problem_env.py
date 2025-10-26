import gymnasium as gym
import numpy as np
from IPython.display import clear_output
import time
from str_generator import StringGenerator

gym.register(
    id="EvenMoreComplexEnv-v0",
    entry_point="even_more_complex_problem_env:EvenMoreComplexEnv",
)

class EvenMoreComplexEnv(gym.Env):
    COMMUNICATE_COST = 15
    
    # symbol replacement
    #    a1 -> a,   c1 -> c
    #   b21 -> x,  b22 -> y,  b23 -> z
    #   e21 -> s,  e22 -> t,  e23 -> r
    
    simulation_transitions={
        1: {'a':2, 'd':3},
        2: {'d':4, 'x':6},
        3: {'a':5},
        4: {'a':2},
        5: {'g':12, 'a':8, 'd':20, 'f':17}, 
        6: {'a':7},
        7: {'c':9},
        8: {'x':10},
        10:{'c':11},
        12:{'a':13},
        13:{'a':14},
        14:{'z':15},
        15:{'r':16},
        16:{'a':5},
        17:{'a':18},
        18:{'y':19},
        19:{'t':16},
        20:{'x':21},
        21:{'s':16}  
    }
    
    po={
        1:  {1,3},
        2:  {2,4,5,12,20,17},
        3:  {6,21},
        4:  {6,10},
        5:  {2,4,8,13,18},
        6:  6,
        7:  7,
        8:  {2,4,14},
        9:  9,
        10: 10,
        11: 11,
        12: {2,4},
        13: {2,4,8,13,18},
        14: 14,
        15: 15,
        16: 16,
        17: {8,13,18},
        19: 19,
        21: 21,
        -1: -1,
    }
    
    global_po_transitions={
        1:{'a':2},
        2:{'x':3, 'a':5},
        3:{'a':7, 's':16},
        4:{'a':7, 'c':11},
        5:{'a':8, 'y':19, 'x':4},
        6:{'a':7},
        7:{'c':9},
        8:{'a':12, 'z':15},
        10:{'c':11},
        12:{'a':12, 'x':6},
        13:{'a':17, 'x':21},
        14:{'z':15},
        15:{'r':16},
        16:{'a':13},
        17:{'x':10, 'y':19, 'a':14,},
        19:{'t':16},
        21:{'s':16},
    }

    local_po_transitions={
        1:{'a':2,                  'c':-1, 'x':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        2:{'x':3, 'a':5,           'c':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        3:{'a':7, 's':16,          'c':-1, 'x':-1, 'y':-1, 'z':-1, 't':-1, 'r':-1},
        4:{'a':7, 'c':11,          'x':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        5:{'a':8, 'y':19, 'x':4,   'c':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        6:{'a':7,                  'c':-1, 'x':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        7:{'c':9,                  'a':-1, 'x':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        8:{'a':12, 'z':15,         'c':-1, 'x':-1, 'y':-1, 's':-1, 't':-1, 'r':-1},
        10:{'c':11,                'a':-1, 'x':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        12:{'a':12, 'x':6,         'c':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        13:{'a':17, 'x':21,        'c':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        14:{'z':15,                'a':-1, 'c':-1, 'x':-1, 'y':-1, 's':-1, 't':-1, 'r':-1},
        15:{'r':16,                'a':-1, 'c':-1, 'x':-1, 'y':-1, 'z':-1, 's':-1, 't':-1},
        16:{'a':13,                'c':-1, 'x':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        17:{'x':10, 'y':19,'a':14, 'c':-1, 'z':-1, 's':-1, 't':-1, 'r':-1},
        19:{'t':16,                'a':-1, 'c':-1, 'x':-1, 'y':-1, 'z':-1, 's':-1, 'r':-1},
        21:{'s':16,                'a':-1, 'c':-1, 'x':-1, 'y':-1, 'z':-1, 't':-1, 'r':-1},
        -1:{'a':-1, 'c':-1, 'x':-1, 'y':-1, 'z':-1, 's':-1, 't':-1, 'r':-1}
    }
    
    metadata = {'render.modes': ['human', 'simulation']}
    
    def __init__(self, render_mode=None, max_star=5):
        self.action_space = gym.spaces.Discrete(2)  # Actions: 0: communicate, 1:don't communicate
        self.observation_space = gym.spaces.Box(low=-1, high=21, shape=(3,), dtype=np.int32)
        
        assert render_mode is None or render_mode in self.metadata['render.modes']
        self.render_mode = render_mode
        
        self.string_generator = StringGenerator(max_star=max_star)
        
    def reset(self, seed=None, options=None):
        # Todo: Generate string
        if self.render_mode == "simulation":
            self.string=self.string_generator.generate_simulation_str()+"$"
        else:
            self.string=self.string_generator.generate_training_str()+"$"
        
        self.string_index = 0
        
        self.global_state = 1
        
        # Initialize agents' partial observations 
        self.agent_1_observation = 1
        self.agent_2_observation = 1
        
        self.communication_count = 0
        
        if self.render_mode == 'human':
            print(f"\nNew Episode: {self.string}")
            self.render()
        
        if self.render_mode == "simulation":
            print(f"\nSimulation String: {self.string}")
            
        if self.string[self.string_index] == 'd':
            self.global_state = self.global_po_transitions[self.global_state].get('d')
        
        config = (self.global_state, self.agent_1_observation, self.agent_2_observation)
        
        info = {"input_alphabet":self.string[self.string_index]}
        
        return np.array(config, dtype=np.int32), info
    
    def step(self, action):
        reward = 0
        agent_id, communicate = action
        terminated = False
        truncated = False
        
        curr_symbol=self.string[self.string_index]
        
        self.global_state = self.global_po_transitions[self.global_state].get(curr_symbol)
        if agent_id==1:
            self.agent_1_observation = self.local_po_transitions[self.agent_1_observation].get(curr_symbol)
            if communicate == 0:
                reward-=self.COMMUNICATE_COST
                self.communication_count+=1
                self.agent_2_observation = self.local_po_transitions[self.agent_2_observation].get(curr_symbol)
        elif agent_id==2:
            self.agent_2_observation = self.local_po_transitions[self.agent_2_observation].get(curr_symbol)
            if communicate == 0:
                self.communication_count+=1
                reward-=self.COMMUNICATE_COST
                self.agent_1_observation = self.local_po_transitions[self.agent_1_observation].get(curr_symbol)
        else:
            raise ValueError("Invalid agent_id. Must be 1 or 2.")
                
        if self.render_mode == 'human':
            print(f"\nAgent {agent_id} {'communicated' if communicate==0 else 'did not communicate'} on '{curr_symbol}'")
            self.render()
        
        self.string_index += 1
        
        curr_symbol=self.string[self.string_index]
        
        if curr_symbol in ['d','g', 'f']:
            self.global_state = self.global_transitions[self.global_state].get(curr_symbol)
            
            if self.render_mode == 'human':
                self.render()
            
            self.string_index += 1
            curr_symbol=self.string[self.string_index]
        
        # reward assignment
        if self.global_state not in [9,11] and self.agent_1_observation in [-1,9,11] and self.agent_2_observation in [-1,9,11]:
            reward -= 50  # both agents think the system is in accepting state but it is not
            terminated = True
        elif self.global_state in [9,11] and (self.agent_1_observation not in [9,11] or self.agent_2_observation not in [9,11]):
            reward -= 50  # one or both agents think the system is not in accepting state but it is
            terminated = True
        elif self.global_state ==9 and self.agent_1_observation ==9 or self.agent_2_observation ==9:
            reward += 50
            terminated = True
        elif self.global_state ==11 and self.agent_1_observation ==11 or self.agent_2_observation ==11:
            reward += 50
            terminated = True
        
        if self.render_mode == "simulation" and self.string[self.string_index]=="$" and self.global_state == 7:
            # condition for simulation where agent successfully disable c at state 7
            truncated = True
        
        config = (self.global_state, self.agent_1_observation, self.agent_2_observation)
        info = {"input_alphabet":self.string[self.string_index]}
        return np.array(config, dtype=np.int32), reward, terminated, truncated, info
        
    def render(self):
        print(f"Current symbol: '{self.string[self.string_index]}'")
        print(f"Config: <{self.global_state}, {self.agent_1_observation}, {self.agent_2_observation}>")  
        print(f"Config: <{self.po[self.global_state]}, {self.po[self.agent_1_observation]}, {self.po[self.agent_2_observation]}>")  
    
    def simulate(self, disabled=True):
        
        
        print(
    "           start             \n"
    "             |               \n"
    "             V               \n"
    "           +-1-+             \n"
   f"           |   |             \n"
    "           +---+             \n"
    "           /   \             \n"
    "        a /     \ d          \n"
    "         V       V           \n"
    "       +-2-+     +-3-+        \n"
   f"   --->|   |     |   |-------- \n"
    "   |   +---+     +---+        |    \n" 
    " a |   /   |                  | a  \n"
    "   |  / d  | x                |    \n"
    "   | V     V                  V \n"
    "  +-4-+  +-6-+              +-5-+\n"
   f"  |   |  |   |       ------>|   |----------\n"
    "  +---+  +---+      /       +---+          \       \n"
    "         /         /       /  |   \         \      \n"
    "      a /         /     g /   | a  \ d       \ f   \n"
    "       V         /       V    V     V         V    \n"
    "   +-7-+        /   +-12-+  +-8-+   +-20-+  +-17-+ \n"
   f"   |   |       |    |    |  |   |   |    |  |    | \n"
    "   +---+       |    +----+  +---+   +----+  +----+ \n"
    "     |         |      |       |       |       |    \n"
    "   c |         |    a |     x |       |       | a  \n"
    "     V         |      V       V       |       V    \n"
    "   +=9=+       |    +-13-+  +-10-+    |     +-18-+ \n"
   f"  ||   ||      |    |    |  |    |    | x   |    | \n"
    "   +===+       |    +----+  +----+    |     +----+ \n"
    "               |      |       |       |       |    \n"
    "               |    a |     c |       |       | y  \n"
    "               |      V       V       V       V    \n"
    "               |    +-14-+  +=11=+  +-21-+  +-19-+ \n"
   f"             a |    |    |  ||  ||  |    |  |    | \n"
    "               |    +----+  +====+  +----+  +----+ \n"
    "               |      |               |       |    \n"
    "               |    z |               |       |    \n"
    "               |      V               | s     |    \n"
    "               |    +-15-+            |       | t  \n"
   f"               |    |    |            |       |    \n"
    "                \   +----+            V       |    \n"
    "                 \      \    r     +-16-+     |    \n"
   f"                  \      --------->|    |<-----    \n"
    "                   \               +----+          \n"
    "                    \_________________|            \n") 