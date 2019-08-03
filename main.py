import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics



df = pd.read_csv('dataset/pmsm_temperature_data.csv')

# seperate the dataset by profile_id as the profile ids are strongly independent
# run only if the files are not generated
if not os.path.exists('dataset/profile_id_4.csv'):
    print('generating csv\'s ...')
    profile_ids = df['profile_id'].unique() # creates a hash table of unique values found in profile_id col

    for p_ids in profile_ids:
        outfilename = 'profile_id_' + str(p_ids) + '.csv'
        print(outfilename)
        df[df['profile_id'] == p_ids].to_csv(os.path.join('dataset', outfilename)) # seperate the dataset according to thier profile ids 

# extract features and lables
df_id_10 = pd.read_csv(os.path.join('dataset', 'profile_id_10.csv'))
y = df_id_10['motor_speed'].values
# ambient temp, coolant temp(motor is water cooled), pm = permanent magnetic surface tempreature (rotor temp), stator yoke temp, stator tooth temp, stator winding temp
X = df_id_10[['ambient', 'coolant','pm', 'stator_yoke', 'stator_tooth', 'stator_winding']] # features that determine the motor speed (the label)


# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

coefs = pd.DataFrame(regressor.coef_, X.columns, columns=['coef'])
coefs.to_csv('coefs.csv')
print("motor_speed = " + str(regressor.coef_) + "X(i)" + " " + str(regressor.intercept_))
print("Accuracy = " + str(regressor.score(X_train, y_train)*100) + "%")


y_pred = regressor.predict(X_test)

ys_df = pd.DataFrame({'Actual':y_test.flatten(), 'Predicted':y_pred.flatten()})
ys_df.to_csv('y_pred.csv')

pred = pd.DataFrame({'actual_motor_speeds':y_test.flatten(),
                    'motor_speeds_predicted':y_pred.flatten(),
                    'ambient':X_test['ambient'], 
                    'coolant':X_test['coolant'],
                    'pm':X_test['pm'], 
                    'stator_yoke':X_test['stator_yoke'], 
                    'stator_tooth':X_test['stator_tooth'], 
                    'stator_winding':X_test['stator_winding']})
pred.to_csv('prediction.csv')
#print(y_pred)
print("Squared error: " + str(metrics.mean_squared_error(y_test, y_pred)))
print("Absolute error: " + str(metrics.mean_absolute_error(y_test, y_pred)))
