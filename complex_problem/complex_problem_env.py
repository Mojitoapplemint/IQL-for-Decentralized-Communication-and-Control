import gymnasium as gym
import numpy as np

gym.register(
    id="ComplexEnv-v0",
    entry_point="complex_env:ComplexEnv",
)

class ComplexEnv(gym.Env):
    
    # symbol replacement
    # a1 -> a
    # a2 -> b
    # e1 -> x
    # e2 -> y
    
    global_transitions={
        0:{'a':1,  'b':2,  'x':0,  'y':0},
        1:{'b':3,  'x':2,  'y':1},
        2:{'a':4},
        3:{'a':4,  'b':1},
        4:{'x':5, 'y':0}
    }
    
    local_transitions={
        0:{'a':1,  'b':2,  'x':0,  'y':0},
        1:{'a':-1, 'b':3,  'x':2,  'y':1},
        2:{'a':4,  'b':-1, 'x':-1, 'y':-1},
        3:{'a':4,  'b':1,  'x':-1, 'y':-1},
        4:{'a':-1, 'b':-1, 'x':5, 'y':0},
        5:{'a':-1, 'b':-1, 'x':-1, 'y':-1},
        -1:{'a':-1, 'b':-1, 'x':-1, 'y':-1},
    }