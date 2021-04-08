from sklearn.linear_model import LinearRegression
import pickle
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import numpy as np
from utils import *

# retrun the linear regression models
def linear_regression(X, y_x, y_y):
    # training x coordinates
    modelx = LinearRegression()
    modelx.fit(X, y_x)
    print('score of x predictor:',modelx.score(X, y_x))

    # training y coordinates
    modely = LinearRegression()
    modely.fit(X, y_y)
    print('score of y predictor:', modely.score(X, y_y))

    return modelx, modely

# retrun the ridge regression models
def ridge_regression(X, y):

    alphas=[1e-2, 1e-4 , 1e-6, 1, 1e2,1e4,1e6, 1e8]

    regressor = RidgeCV(alphas=alphas, store_cv_values=True)
    regressor.fit(X, y)
    cv_mse = np.mean(regressor.cv_values_, axis=0)


    print(alphas)
    print(cv_mse)

    return  regressor

# returns the trained model based on the data in the dataset_path, and given the type of the model
def get_trained_models(X, y, type = "ridge", save_name = "default", save = True):
    y_x, y_y = [], []
    for (x, y) in y:
        y_x.append(x)
        y_y.append(y)

    if (type=="linear"):
        modelx, modely = linear_regression(X, y_x, y_y)
    if (type=="ridge"):
        modelx = ridge_regression(X, y_x)
        modely = ridge_regression(X, y_y)

    if save:
        # save the model to disk
        filename = 'models/modelx_'+save_name+'.sav'
        pickle.dump(modelx, open(filename, 'wb'))

        # save the model to disk
        filename = 'models/modely_'+save_name+'.sav'
        pickle.dump(modely, open(filename, 'wb'))

    return modelx, modely

def testModel(X, y, modelx, modely):
    y_x, y_y = [], []
    for (x, y) in y:
        y_x.append(x)
        y_y.append(y)

    predx = modelx.predict(X)
    predy = modely.predict(X)

    sum = 0
    count = len(y_x)
    for i in range(count):
        sum += distance2D(y_x[i], y_y[i], predx[i], predy[i])

    return sum / count

if __name__ == "__main__":

    # training and saving the model
    dataset_path = "pictures/stable2"
    # X is the images, y_x,y_y is the coordinates respectively
    X, y_x, y_y = load_dataset(dataset_path)
    print("shape of dataset: ", len(X))

    y = []
    for i in range (len(y_x)):
        y.append((y_x[i], y_y[i]))


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 60)

    modelx, modely = get_trained_models(X_train, y_train, type="linear", save_name="ridge_stable2", save=True)

    error = testModel(X_test, y_test, modelx, modely)
    print("error in pixels:", error)

