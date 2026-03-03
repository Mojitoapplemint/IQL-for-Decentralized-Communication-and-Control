
import numpy as np
import random

class WordGenerator:
    def __init__(self, max_star=5):
        self.max_star = max_star

    def generate_simulation_word(self):

        return random.choice(['cax', 'cay', 'acx', 'acy']) + 's'
    
    def generate_training_word(self):
        return random.choice(['cax', 'cay', 'acx', 'acy']) + 's'

# import pandas as pd
# generator = WordGenerator(max_star=5)

# testing_pool = []

# count = 0

# while count<1000:
#     word = generator.generate_simulation_word()
#     if word not in testing_pool:
#         testing_pool.append(word)
#     count+=1

# testing_pool_df = pd.DataFrame(testing_pool, columns=['word'])

# testing_pool_df.to_csv('three_agents_long_short/words_for_stats.csv', index=False)