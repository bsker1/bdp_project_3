import numpy as np
import scipy as sp
import pandas as pd
import statsmodels as sm

def main():
    # Read Data frame

    Trials = 100
    Contestants = 50
    StartingCof = 0.5 # 1 is AI 0 is human
    Factors = 1 # Update for every factor

    HumanOrAI = np.array() # Where the size is 1 X Contenstants
    Data = np.array() # Obs X Factor where Obs is that number of Contestants and Factor is the amount of factors 

    Weights = list()
    for X in range(Factors):
        Weights.append(StartingCof)

    for T in range(Trials):
        #reapply randomness to 50 duplicants
        for C in range(Contestants):
            print("Remove This Line")
            #Make 50 contestants
            #randomize the weight 
                       
        #Read all 50 models find the best
        #Set that to the base model