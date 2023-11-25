import numpy as np
import pandas as pd
import re
from readability import Readability

def main():
    EssayFrame = pd.read_csv("../sampled_data.csv")

    FactorFrame = EssayFrame.copy(deep=False)

    #Data cleaning
    #Thanks to chatgpt for this wonderful 1 liner to drop all essays with less than 50 words
    FactorFrame.drop(FactorFrame[FactorFrame["text"].apply(lambda x: len(x.split()) < 50)].index)

    # Get Word Count
    FactorFrame.insert(column="WordCount", loc=2, value=0)
    for pos, row in FactorFrame.iterrows():
        FactorFrame.at[pos, "WordCount"] = len(row["text"].split())

    # Get pronoun Count
    Pronouns = re.compile(r'\b(I|me|myself|mine|my|we|us|ourselves|ours|our|Me|Myself|Mine|My|We|Us|Outselves|Ours|Our)\b')
    FactorFrame.insert(column="PronounCount", loc=3, value=0)
    for pos, row in FactorFrame.iterrows():
        FactorFrame.at[pos, "PronounCount"] = len(Pronouns.findall(row["text"]))

    # Get Dale-Chall Readabilty
    # import nltk
    # nltk.download()
    # Run the commands above in the python terminal and download all
    FactorFrame.insert(column="Readabilty", loc=4, value=0)
    for pos, row in FactorFrame.iterrows():
        FactorFrame.at[pos, "Readabilty"] = Readability(row["text"]).dale_chall().score


    #Print Dataframe
    print(FactorFrame)

    #Make sure to write new dataframe
    FactorFrame.to_csv("../FactorizedData.csv")
    
if __name__ == '__main__':
    main()
