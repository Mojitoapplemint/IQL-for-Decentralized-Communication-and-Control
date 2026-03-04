import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
import sys
sys.path.insert(0, './problem_w_unobservable_events')
import uo_problem_env
from word_generator import WordGenerator

protocol_with_returns = pd.read_csv("problem_w_unobservable_events/larger_11_penalty_successful_protocols_with_returns.csv")

returns = protocol_with_returns["Agent 2 Average Cumulative Reward"].unique()

for return_value in returns:
    protocol_with_specific_return = protocol_with_returns[protocol_with_returns['Agent 2 Average Cumulative Reward'] == return_value]
    protocol_with_specific_return.drop(columns=["Agent 1 Average Cumulative Reward","Agent 2 Average Cumulative Reward"]).to_csv(f"problem_w_unobservable_events/larger_11_penalty_successful_protocols_with_{return_value}_return.csv", index=False)
