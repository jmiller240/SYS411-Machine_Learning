# Multivariate regression with train and test
# Jack Miller
# Daniel Gibson
# February, 2022
import numpy as np
import pandas as pd


# Read data from file using pandas and create a dataframe
housingDF = pd.read_csv('housing.csv')

# Minimizes JCost and reports percent error
# Returns thetas, percent error
def normalEQLinReg(features, df, thetas):

    # Subdivide the data into features (Xs) and dependent variable (y) dataframes
    XsDF = df[features]
    YDF = df['price']

    # Convert dataframes to numpy ndarray(matrix) types
    Xs = XsDF.to_numpy()
    Y = YDF.to_numpy()

    # Add the 1's column to the Xs matrix (1 * the intercept values)
    XsRows, XsCols = Xs.shape
    X0 = np.ones((XsRows, 1))
    Xs = np.hstack((X0, Xs))

    # Calc the Thetas via the normal equation if we're training the data; otherwise, skip
    if len(thetas) == 0:
        thetas = (np.linalg.pinv(Xs.T @ Xs)) @ Xs.T @ Y
    # print(f"thetas: {thetas}")

    # Now, generate differences from the predicted
    predictedM = (Xs @ thetas.T)
    diffs = abs(predictedM - Y)
    sumOfDiffs = diffs.sum()
    sumOfPrices = Y.sum()
    avDiff = round(sumOfDiffs / sumOfPrices * 100, 1)
    # print("training features:", features)
    # print("average price difference for training values:", str(avDiff) + "%" + "\n")
    return thetas, avDiff


# Runs an experiment (sampling, training, and testing) n times and reports the average performance in percent error
def runExperiment(features, n):
    totalPercentError = 0

    for i in range(n):
        trainingDF = housingDF.sample(frac=0.7)
        testingDF = housingDF[~housingDF.index.isin(trainingDF.index)]

        thetas, percentError = normalEQLinReg(features, trainingDF, [])
        thetas, percentError = normalEQLinReg(['sqft_living', 'condition', 'yr_built', 'sqft_lot', 'floors', 'view'], testingDF, thetas)
        print(i, ": ", thetas)
        totalPercentError += percentError

    print(f"Average percent error for testing data with {n} experiments:", str((totalPercentError/n)) + "%" + "\n")


runExperiment(['sqft_living', 'condition', 'yr_built', 'sqft_lot', 'floors', 'view'], 100)


