import numpy as np
import pandas as pd
import re
from readability import Readability

def main():
    FileNames = ["../sampled_data.csv", "../training_data.csv"]

    for Name in FileNames:

        EssayFrame = pd.read_csv(Name)

        FactorFrame = EssayFrame.copy(deep=False)

        #Data cleaning
        #Thanks to chatgpt for this wonderful 1 liner to drop all essays with less than 100 words
        FactorFrame.drop(FactorFrame[FactorFrame["text"].apply(lambda x: len(x.split()) < 101)].index)

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

        # Get Paragraph Count
        FactorFrame.insert(column="ParagraphCount", loc=5, value=0)
        for pos, row in FactorFrame.iterrows():
            FactorFrame.at[pos, "ParagraphCount"] = len(row["text"].split("\n"))

        # Get Paragraph size choesion (Standard Deviation)
        FactorFrame.insert(column="ParagraphSizeCohesion", loc=6, value=0)
        for pos, row in FactorFrame.iterrows():
            ParagraphList = row["text"].split("\n")
            ParagraphSizes = []
            for Paragraph in ParagraphList:
                ParagraphSizes.append(len(Paragraph))
            FactorFrame.at[pos, "ParagraphSizeCohesion"] = np.std(ParagraphSizes)

        #Print Dataframe
        print(FactorFrame)

        #Make sure to write new dataframe
        prefix = Name[:-4]
        file = prefix + "_Factorized.csv"
        FactorFrame.to_csv(file)
    
if __name__ == '__main__':
    main()
