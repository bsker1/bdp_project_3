import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import sklearn as sk
from sklearn.linear_model import LogisticRegression

def main(rng):
    TrainingFrame = pd.read_csv("../FactorizedData.csv")

    #endog
    ResponseCollection = TrainingFrame["label"]

    #exog
    Input = TrainingFrame[["WordCount", "PronounCount", "Readabilty", "ParagraphCount", "ParagraphSizeCohesion"]].to_numpy()

    #Statsmodel Logistic regression (needed to classifiy binary outputs)
    model = sm.Logit(ResponseCollection, Input)
    results = model.fit()
    print(results.summary())

    Xtrain = Input
    Xtest = None #Get a dataframe of factorized test data
    Ytrain = ResponseCollection
    Ytest = None #Get a series of labels of Xtest data

    Contestants = 10
    StartWeight = None #Create a dict of labels and weights constant
    #Apply randomness
    #Run test selected the best R^2 and redo randomness

if __name__ == '__main__':
    rng = np.random.default_rng()
    main(rng)