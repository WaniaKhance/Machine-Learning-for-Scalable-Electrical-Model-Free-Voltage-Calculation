import pandas as pd                                         # for working with DataFrames
import sqlite3                                              # for working with SQLite DB
import numpy as np                                          # for array operations
import random                                               # for generating random numbers
import matplotlib.pyplot as plt                             # for data visualization
from sklearn.model_selection import train_test_split        # for splitting the data
from sklearn.metrics import mean_squared_error              # for calculating the cost function
from sklearn.ensemble import RandomForestRegressor          # for building the model
from xgboost import XGBRegressor        
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
#!pip install tensorflow
#!pip install keras
#from keras.models import Sequential
#from keras.layers import Dense


def data_generation_with_high_demand(net_db):

    meter_profiles_df = pd.read_sql_query("SELECT DISTINCT Meter.ID FROM Meter", net_db)        # Reading Meters IDs from DB 
    meter_profiles = meter_profiles_df['ID'].values.tolist()                                    # Adding into a list from dataframe
    randomlist = random.choices(range(1, 6), k = 266)                                           # Generate list of 266 random numbers between 1 and 5 same
    for i,j in zip(randomlist,meter_profiles):
        df_meter = pd.read_csv('PQ_csv/P{}.csv'.format(j), header=None)                         # Reading CSV file of a Meter Profile j
        df_meter = df_meter.mul(i)                                                              # Multiplying the list of random numbers to 
        df_meter.to_csv('New_PQ_csv/P{}.csv'.format(j), index = False, header=None)             # Saving new CSV files to a folder
    

def pq_data_aggregation(agg_df):

    bus = agg_df.Bus.unique()
    for i in bus:    # i gives Bus ID

        Phase_A_DF =  pd.DataFrame(np.zeros((744, 2)))                             # Creating dataframe of 2 columns and 743 rows with zeros
        Phase_A_DF.columns = ['P', 'Q']
        Phase_B_DF =  pd.DataFrame(np.zeros((744, 2)))                             # Creating dataframe of 2 columns and 743 rows with zeros
        Phase_B_DF.columns = ['P', 'Q']
        Phase_C_DF =  pd.DataFrame(np.zeros((744, 2)))                             # Creating dataframe of 2 columns and 743 rows with zeros
        Phase_C_DF.columns = ['P', 'Q']
        Phase_D_DF =  pd.DataFrame(np.zeros((744, 2)))                             # Creating dataframe of 2 columns and 743 rows with zeros
        Phase_D_DF.columns = ['P', 'Q']

        df_meter_bus = agg_df[agg_df['Bus'] == i]                                                             # Give dataframe of all Meters within Bus i
        df_meter_bus_ph1 = df_meter_bus[df_meter_bus['Phase'] == 1]                                           # Get only Meters with phase 1 out of all Meters of Bus i
        
        for j in df_meter_bus_ph1['Meter']:                                                                   # Run a loop on each meter that are in phase 1
            df_meter_bus_ph1_PQ = pd.read_csv('New_PQ_csv/P{}.csv'.format(j), header=None)      # Reading CSV file of a Meter Profile j
            df_meter_bus_ph1_PQ.columns = ['P', 'Q']
            Phase_A_DF = Phase_A_DF.add(df_meter_bus_ph1_PQ, fill_value=0)

        df_meter_bus_ph2 = df_meter_bus[df_meter_bus['Phase'] == 2]                                           # Get only Meters with phase 2 out of all Meters of Bus i

        for j in df_meter_bus_ph2['Meter']:

            df_meter_bus_ph2_PQ = pd.read_csv('New_PQ_csv/P{}.csv'.format(j), header=None)      # Reading CSV file of a Meter Profile j
            df_meter_bus_ph2_PQ.columns = ['P', 'Q']
            Phase_B_DF = Phase_B_DF.add(df_meter_bus_ph2_PQ, fill_value=0)

        df_meter_bus_ph3 = df_meter_bus[df_meter_bus['Phase'] == 3]                                           # Get only Meters with phase 2 out of all Meters of Bus i

        for j in df_meter_bus_ph3['Meter']:

            df_meter_bus_ph3_PQ = pd.read_csv('New_PQ_csv/P{}.csv'.format(j), header=None)      # Reading CSV file of a Meter Profile j
            df_meter_bus_ph3_PQ.columns = ['P', 'Q']
            Phase_C_DF = Phase_C_DF.add(df_meter_bus_ph3_PQ, fill_value=0)

        df_meter_bus_ph4 = df_meter_bus[df_meter_bus['Phase'] == 4]                                           # Get only Meters with phase 4 out of all Meters of Bus i

        for j in df_meter_bus_ph4['Meter']:

            df_meter_bus_ph4_PQ = pd.read_csv('New_PQ_csv/P{}.csv'.format(j), header=None)      # Reading CSV file of a Meter Profile j
            df_meter_bus_ph4_PQ.columns = ['P', 'Q']
            Phase_D_DF = Phase_D_DF.add(df_meter_bus_ph4_PQ, fill_value=0)

        Phase_D_DF = Phase_D_DF/3                                     # Dividing whole Phase 4 dataframe with 3
        Phase_A_DF = Phase_A_DF.add(Phase_D_DF, fill_value=0)        # Adding 1/3 of Phase D to Phase A dataframe
        Phase_B_DF = Phase_B_DF.add(Phase_D_DF, fill_value=0)        # Adding 1/3 of Phase D to Phase B dataframe
        Phase_C_DF = Phase_C_DF.add(Phase_D_DF, fill_value=0)        # Adding 1/3 of Phase D to Phase C dataframe

        Phase_A_DF.columns = ['Pa', 'Qa']                            # Renaming Column names from P and Q to Pa and Qa
        Phase_B_DF.columns = ['Pb', 'Qb']                            # Renaming Column names from P and Q to Pb and Qb
        Phase_C_DF.columns = ['Pc', 'Qc']                            # Renaming Column names from P and Q to Pc and Qc

        final_df = pd.concat([Phase_A_DF, Phase_B_DF, Phase_C_DF], axis=1)      # axis = 1 shows concatenation along columns
        final_df = final_df.round(5)
        final_df.to_csv('BUS PQ FINAL/Bus_{}_PQ.csv'.format(i), index = False)
        

def volts_data_aggregation(agg_df):

    voltage_df = pd.DataFrame()
    values = agg_df.Bus.unique()

    for i in values:        
        volts = pd.read_csv('/gdrive/My Drive/Pablo project/out/TestNet_Mon_b{}_1.csv'.format(i))       #Reading Voltages file for Bus ID i.
        volts.columns = volts.columns.str.strip()                                                       # Column name has dot like V1.re which causes problem in reading, this command solves it.
        v1_mag = np.sqrt((volts['V1.re']- volts['V4.re'])**2 + (volts['V1.im']- volts['V4.im'])**2)
        v2_mag = np.sqrt((volts['V2.re']-volts['V4.re'])**2 + (volts['V2.im']-volts['V4.im'])**2)
        v3_mag = np.sqrt((volts['V3.re']-volts['V4.re'])**2 + (volts['V3.im']-volts['V4.im'])**2)
        voltage_df['V1_Magnitude'] = ((np.sqrt(3)) * (v1_mag/ 400.0))
        voltage_df['V2_Magnitude'] = ((np.sqrt(3)) * (v2_mag/ 400.0))
        voltage_df['V3_Magnitude'] = ((np.sqrt(3)) * (v3_mag/ 400.0))
        voltage_df.to_csv('BUS VOLTS FINAL/Bus_{}_Volts.csv'.format(i), index = False)
        

def complete_data_aggregation(agg_df):

    values = agg_df.Bus.unique()
    for i in values:    # i gives Bus ID
        PQ_df = pd.read_csv('BUS PQ FINAL/Bus_{}_PQ.csv'.format(i))
        Voltage_df = pd.read_csv('BUS VOLTS FINAL/Bus_{}_Volts.csv'.format(i)) #, skipfooter = 1
        aggregated_df = pd.concat([PQ_df, Voltage_df], axis=1)
        aggregated_df.to_csv('AGG_Data/Bus_{}_PQ_Volts.csv'.format(i), index = False)
        

def time_based_aggregation(agg_df):

    buses = agg_df.Bus.unique()

    final_df = pd.DataFrame(columns=['Hour', 'Bus', 'Pa', 'Qa', 'Pb', 'Qb', 'Pc', 'Qc', 'V1_Magnitude','V2_Magnitude', 'V3_Magnitude'])
    temp_df = pd.DataFrame(columns=['Hour', 'Bus'])
    temp_df2 = pd.DataFrame(columns=['Pa', 'Qa', 'Pb', 'Qb', 'Pc', 'Qc', 'V1_Magnitude','V2_Magnitude', 'V3_Magnitude'])
    abc = pd.DataFrame()

    for i in range (744):                                                               # For each time instance i.e. 1 to 744
        for j in buses:                                                                 # Reading each bus one by one
            df = pd.read_csv('AGG_Data/Bus_{}_PQ_Volts.csv'.format(j))                   # Reading aggregated data of PQ and voltages of all meters related to above one bus
            temp_df[['Hour', 'Bus']] = [[i, j]]
            temp_df2 = df.iloc[[i]]                                                      # Creating another temparory dataframe which has the all the data of both PQ and voltages for 'i' specific time instance
            temp_df2 = temp_df2.reset_index(drop=True)
            abc = pd.concat([temp_df.iloc[[0]], temp_df2.iloc[[0]]], axis = 1)           # Concatinating data of both temparory dataframes which includes Hour, Bus, P, Q, Voltages values.
            final_df = final_df.append(abc)                                              # Appending all data to a final dataframe which will be saved after completing all buses and time instances
            final_df = final_df.reset_index(drop=True)
            abc = pd.DataFrame()
            temp_df2 = pd.DataFrame()
    
    final_df.to_csv('Time_Agg_Final.csv', index = False)
    dataframe = pd.read_csv('Time_Agg_Final.csv')                                        # Doing it to increment hour by 1 as it starts from 0
    dataframe['Hour'] = dataframe['Hour'] + 1
    dataframe.to_csv('Time_Agg_Final.csv', index = False)                                # Saving it to the same file


def modeling_with_random_forest_regressor(x, y):

    # Splitting the dataset into training and testing set (80/20)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 32)
    # Initializing the Random Forest Regression model with 5 decision trees
    model = RandomForestRegressor(n_estimators = 5, random_state = 4)
    # Fitting the Random Forest Regression model to the data
    model.fit(x_train, y_train) 
    y_pred = model.predict(x_test)
    # RMSE (Root Mean Square Error)
    rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
    #print("\n Overall RMSE: ", rmse)
    return rmse


def modeling_with_XGBoost_regressor(x, y):
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 32)
    #RegModel=XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=180, objective='reg:linear', booster='gbtree')
    RegModel = XGBRegressor(n_estimators=100, max_depth=5, eta=0.3, subsample=0.7, colsample_bytree=1)
    XGB = RegModel.fit(x_train,y_train)
    prediction = XGB.predict(x_test)
    r_rmse = np.sqrt(mean_squared_error(y_test, prediction))
    return r_rmse


def modeling_with_ANN(x, y):
    PredictorScaler=StandardScaler()
    TargetVarScaler=StandardScaler()
    ### Sandardization of data ###
    # Storing the fit object for later reference
    PredictorScalerFit=PredictorScaler.fit(x)
    TargetVarScalerFit=TargetVarScaler.fit(y)
    # Generating the standardized values of X and y
    X = PredictorScalerFit.transform(x)
    Y = TargetVarScalerFit.transform(y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=32)
    # create ANN model
    model = Sequential()
    # Defining the Input layer and FIRST hidden layer, both are same!
    model.add(Dense(units=5, input_dim=6, kernel_initializer='normal', activation='relu'))
    # Defining the Second layer of the model
    # after the first layer we don't have to specify input_dim as keras configure it automatically
    model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))
    # The output neuron is a single fully connected node
    model.add(Dense(3, kernel_initializer='normal'))
    # Compiling the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    # Fitting the ANN to the Training set
    model.fit(x_train, y_train ,batch_size = 20, epochs = 5, verbose=1)
    predictions = model.predict(x_test)
    # Scaling the predicted Price data back to original price scale
    predictions = TargetVarScalerFit.inverse_transform(predictions)
    # Scaling the y_test Price data back to original price scale
    y_test_orig = TargetVarScalerFit.inverse_transform(y_test)
    return mean_absolute_error(y_test_orig, predictions) 



if __name__ == "__main__":

    net_db = sqlite3.connect("net.db")
    agg_df = pd.read_sql_query("SELECT CGP.Bus, Meter.ID, Meter.Phase1 FROM CGP, Meter WHERE CGP.ID = Meter.CGP1", net_db)
    agg_df.rename(columns = {'ID':'Meter', 'Phase1':'Phase'}, inplace = True)
    
    # Data generation and aggregation

    # Following lines of code should be run once, as it will generate data again  
    # PQ_csv folder is already incremented so no need to do it again and Final data is already ready to use.  

    data_generation_with_high_demand(net_db)                   # For data regeneration with high demand
    pq_data_aggregation(agg_df)                                # For PQ data aggregation from regenerated data 
    volts_data_aggregation(agg_df)                             # For voltage calculation 
    complete_data_aggregation(agg_df)                          # To perform PQ and Volts aggregation for each Bus 
    time_based_aggregation(agg_df)                             # Aggregating all buses on hourly basis

    # Data Modelling and prediction results

    dataframe = pd.read_csv('Time_Agg_Final.csv')
    input_features = dataframe[['Pa', 'Qa', 'Pb', 'Qb','Pc','Qc']]                                    
    output_features = dataframe[['V1_Magnitude','V2_Magnitude','V3_Magnitude']]                       
    
    RF_RMSE  = modeling_with_random_forest_regressor(input_features, output_features)
    print("Root Mean Square Error of Random Forest Regression: ", RF_RMSE)
    XGB_RMSE = modeling_with_XGBoost_regressor(input_features, output_features)
    print("Root Mean Square Error of XGBoost Regression: ", XGB_RMSE)
    ANN_MAE = modeling_with_ANN(input_features, output_features)
    print('Mean Absolute Error for ANN: ', ANN_MAE)