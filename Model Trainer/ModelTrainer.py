import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main(rng):
    TrainingFrame = pd.read_csv("../training_data_Factorized.csv")
    TestFrame = pd.read_csv("../sampled_data_Factorized.csv")

    #Training Sets
    ResponseCollection = TrainingFrame["label"]
    Input = TrainingFrame[["WordCount", "PronounCount", "Readabilty", "ParagraphCount", "ParagraphSizeCohesion"]].to_numpy()

    #Test Sets
    TestResponse = TestFrame["label"]
    TestInput = TestFrame[["WordCount", "PronounCount", "Readabilty", "ParagraphCount", "ParagraphSizeCohesion"]].to_numpy()

    Xtrain = Input
    Xtest = TestInput
    Ytrain = ResponseCollection
    Ytest = TestResponse

    print("Stats model regression stats")
    print("////////////////////////////")
    model = sm.Logit(ResponseCollection, Input)
    results = model.fit()
    print(results.summary())

    print("/////////////////////////////")


    #Scikit logestic regression (Balanced)
    model = LogisticRegression()
    model.fit(Xtrain, Ytrain)

    #Scikit auto adjusting weight regression
    Contestants = 10
    StandardStartWeight = {"WordCount": 0.5, "PronounCount": 0.5, "Readabilty":0.5, "ParagraphCount":0.5, "ParagraphSizeCohesion": 0.5, 0:1, 1:12}
    StartWeightList = [StandardStartWeight]

    for Trails in range(5):
        for ContestantNo in range(Contestants - 1):
            RngList = {}
            for key in StartWeightList[0].keys():
                RngList[key] = StandardStartWeight[key] + StandardStartWeight[key] * np.random.uniform(low = -0.2, high = 0.2)
            StartWeightList.append(RngList)

        AccuracyList = []

        for Contestant in StartWeightList:
            regression = LogisticRegression(class_weight=Contestant, max_iter=100)
            regression.fit(Xtrain, Ytrain)
            predictions = regression.predict(Xtest)
            accuracy = accuracy_score(Ytest, predictions)
            AccuracyList.append(accuracy)


        BestIndex = AccuracyList.index(max(AccuracyList))
        StandardStartWeight = StartWeightList[BestIndex]
        print("Accuracy is now at :")
        print(AccuracyList[BestIndex])
        print("Weight is")
        print(StartWeightList[BestIndex])

    WeightedAdjusterScore = regression.score(Xtest, Ytest)
    BalancedWeight = model.score(Xtest, Ytest)

    print("Weight adjuster scored at:")
    print(WeightedAdjusterScore)
    print("Balanced weight scored at:")
    print(BalancedWeight)

if __name__ == '__main__':
    rng = np.random.default_rng()
    main(rng)