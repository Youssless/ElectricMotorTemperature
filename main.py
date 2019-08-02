import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


def main():
    df = pd.read_csv('dataset/pmsm_temperature_data.csv')
    
    # seperate the dataset by profile_id as the profile ids are strongly independent
    #profile_ids = df['profile_id'].unique()
    #outfilenames = []
    '''
    for p_ids in profile_ids:
        outfilename = 'profile_id_' + str(p_ids) + '.csv'
        print(outfilename)
        df[df['profile_id'] == p_ids].to_csv(os.path.join('dataset', outfilename))
        outfilenames.insert(p_ids, outfilename)
    '''
    df_id_4 = pd.read_csv(os.path.join('dataset', 'profile_id_4.csv'))
    y = df_id_4['motor_speed'].values.reshape(-1, 1)
    X = df_id_4[['ambient', 'coolant','pm', 'stator_yoke', 'stator_tooth', 'stator_winding']].values.reshape(-1, 6)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    print(regressor.intercept_)
    print(regressor.coef_)
    print(regressor.score(X_train, y_train))
    y_pred = regressor.predict(X_test)
    print(y_pred)
    print(metrics.mean_squared_error(y_test, y_pred))





if __name__ == '__main__':
    main()