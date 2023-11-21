import numpy as np
import scipy as sp
import pandas as pd
import statsmodels as sm

def main():
    EssayFrame = pd.read_csv("../sampled_data.csv")

    FactorFrame = EssayFrame.copy(deep=False)

    #Data cleaning
    #Thanks to chatgpt for this wonderful 1 liner to drop all essays with less than 50 words
    FactorFrame.drop(FactorFrame[FactorFrame["text"].apply(lambda x: len(x.split()) < 50)].index)

    # Get Word Count
    FactorFrame.insert(column="WordCount", loc=2, value=0)
    for pos, row in FactorFrame.iterrows():
            FactorFrame.at[pos, "WordCount"]= len(row["text"].split())

    #Print Dataframe
    print(FactorFrame)

    #Make sure to write new dataframe
    FactorFrame.to_csv("../FactorizedData.csv")
    
if __name__ == '__main__':
    main()
