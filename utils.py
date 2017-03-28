import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Lasso
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

le = LabelEncoder()



#impure factorize function
def factorize(data_frame, factorize_data_frame, column, fill_na = None ):
    factorize_data_frame[column] = data_frame[column]
    if fill_na is not None:
        factorize_data_frame[column].fillna(fill_na,inplace=True)
    le.fit(factorize_data_frame[column].unique())
    factorize_data_frame[column] = le.transform(factorize_data_frame[column])
    return factorize_data_frame

#impure converter categorical features
def refactor(refactored_data_frame,data_frame,column, fill_na):
    refactored_data_frame[column] = data_frame[column]
    if fill_na is not None:
        refactored_data_frame.fillna(fill_na,inplace=True)

    dummies = pd.get_dummies(refactored_data_frame[column],prefix="_" +column)
    refactored_data_frame = refactored_data_frame.join(dummies)
    refactored_data_frame = refactored_data_frame.drop([column], axis = 1)
    return refactored_data_frame

def lasso(train ,test , label, alpha = 0.00099, max_iteration = 50000):
    lasso = Lasso(alpha = alpha , max_iter = max_iteration)
    lasso.fit(train,label)

    #prediction on training data
    y_predicton = lasso.predict(train)
    y_test = label
    print("Lasso score on training set: ", rmse(y_test, y_predicton))

    y_predicton = lasso.predict(test)
    y_predicton = np.exp(y_predicton)
    return y_predicton

def gradient_boosting(train ,test ,label):
    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                    min_samples_leaf=15, min_samples_split=10, loss='huber')
    gb.fit(train,label.as_matrix().ravel())

    # prediction on training data
    y_predicton = gb.predict(train)
    y_test = label
    print("Gradient Boosting score on training set: ", rmse(y_test, y_predicton))

    y_prediction = gb.predict(test)
    y_prediction = np.exp(y_prediction)
    return y_prediction

def random_forest(train ,test ,label):
    rf = RandomForestRegressor(n_estimators=150, n_jobs=4)
    rf.fit(train,label.as_matrix().ravel())

    # prediction on training data
    y_predicton = rf.predict(train)
    y_test = label
    print("Random Forest score on training set: ", rmse(y_test, y_predicton))
    y_predicton = rf.predict(test)
    y_predicton = np.exp(y_predicton)
    return y_predicton


def rmse(y,y_prediction):
    return np.sqrt(mean_squared_error(y,y_prediction))


