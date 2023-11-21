import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import sklearn as sk
from sklearn.linear_model import LogisticRegression

def main(rng):
    TrainingFrame = pd.read_csv("../FactorizedData.csv")

    Trials = 1
    Contestants = 10
    StartingCof = 0.5 # 1 is AI 0 is human
    Factors = 1 # Update for every factor

    HumanOrAI = np.array((Contestants + 1)) # Where the size is 1 X Contenstants
    Data = np.array((Contestants, Factors)) # Obs X Factor where Obs is that number of Contestants and Factor is the amount of factors

    #endog
    ResponseCollection = TrainingFrame["label"]

    #exog
    Input = TrainingFrame["WordCount"]

    # weights unused will be used on weighted LogisticRegression in scikit
    Weights = list()
    for X in range(Contestants):
        Weights.append(StartingCof + rng.random())

    #Statsmodel Logistic regression (needed to classifiy binary outputs)
    model = sm.Logit(ResponseCollection, Input)
    results = model.fit()
    print(results.summary())

    LogReg = LogisticRegression()
    LogReg.fit(Input, ResponseCollection)

    #Insert a test set to predict if it AI or not
    #LogReg.predict()


if __name__ == '__main__':
    rng = np.random.default_rng()
    main(rng)