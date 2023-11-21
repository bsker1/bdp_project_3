import pandas as pd
import numpy as np

def generate_indicies(in_df, in_size):
    used_indicies = np.empty(shape=in_size, dtype=int)
    df_size = in_df.count().iloc[0]

    index = 0
    while index < in_size:
        # pick random entry in df
        random_index = np.random.randint(low=0, high=df_size, dtype=int)

        # ensure no duplicate entries are picked
        no_duplicates = True
        for i in range(index):
            if random_index == used_indicies[i]:
                no_duplicates = False
                break

        # iterate if entry hasn't been picked
        if no_duplicates:
            used_indicies[index] = random_index
            index += 1
    
    return used_indicies



# separate data into AI vs human
zero_df = pd.read_csv('../raw_data.csv').query('label == 0')
one_df = pd.read_csv('../raw_data.csv').query('label == 1')

# pick 50 AI excerpts, 50 human
zero_indicies = generate_indicies(zero_df, 50)
one_indicies = generate_indicies(one_df, 50)

# write subset_df to new csv file
subset_df = pd.DataFrame(columns=['text', 'label'])
for i in range(50):
    subset_df.loc[i] = [zero_df.iloc[zero_indicies[i]].iloc[0], '0']
for i in range(50):
    subset_df.loc[i + 50] = [one_df.iloc[one_indicies[i]].iloc[0], '1']

subset_df.to_csv('../sampled_data.csv', encoding='utf-8', index=False)