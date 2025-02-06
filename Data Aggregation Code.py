#Importing packages/libraries

import pandas as pd
import sqlite3
from pandasql import sqldf
import numpy as np



# Read sqlite query results into a pandas DataFrame

con = sqlite3.connect("/Users/wania/Desktop/Project/PLX_65019/net.db")




CGP_df = pd.read_sql_query("SELECT CGP.Bus, CGP.ID from CGP ORDER BY CGP.Bus", con)#.drop(index=0)

CGP_df.rename(columns = {'ID':'CGPID'}, inplace = True)         # In CGP DB, Renaming column ID to CGPID

#print(CGP_df) # print total 101 rows x 2 columns



Meter_df = pd.read_sql_query("SELECT Meter.ID, Meter.Phase1, Meter.CGP1 from Meter ", con)  

Meter_df.rename(columns = {'CGP1':'CGPID'}, inplace = True)         #  In Meter DB, Renaming Column CGP1 to CGPID

Meter_df.rename(columns = {'ID':'Meter_ID'}, inplace = True)        #  Renaming Column ID to Meter_ID for easy understanding

#print(Meter_df)  #  print 266 rows x 3 columns



#Combining CGP DB and Meter DB dataframes based on similar CGPID, give all the rows of meter_df

combined_df = Meter_df.merge(CGP_df, on = 'CGPID', how = 'inner')

#print(combined_df)



#  Aggregation of data

Phase_A_DF =  pd.DataFrame(np.zeros((743, 2)))                             # Creating dataframe of 2 columns and 743 rows with zeros
Phase_A_DF.columns = ['P', 'Q']
Phase_B_DF =  pd.DataFrame(np.zeros((743, 2)))                             # Creating dataframe of 2 columns and 743 rows with zeros
Phase_B_DF.columns = ['P', 'Q']
Phase_C_DF =  pd.DataFrame(np.zeros((743, 2)))                             # Creating dataframe of 2 columns and 743 rows with zeros
Phase_C_DF.columns = ['P', 'Q']
Phase_D_DF =  pd.DataFrame(np.zeros((743, 2)))                             # Creating dataframe of 2 columns and 743 rows with zeros
Phase_D_DF.columns = ['P', 'Q']

for i in CGP_df.Bus:    # i gives Bus ID


    df_meter_bus = combined_df[combined_df['Bus'] == i]             # Give dataframe of all Meters within Bus i 

    df_meter_bus_ph1 = df_meter_bus[df_meter_bus['Phase1'] == 1]    # Get only Meters with phase 1 out of all Meters of Bus i 

    for j in df_meter_bus_ph1['Meter_ID']:                          # Run a loop on each meter that are in phase 1
        
        df_meter_bus_ph1_PQ = pd.read_csv('/Users/wania/Desktop/Project/PLX_65019/PQ_csv/P{}.csv'.format(j))      # Reading CSV file of a Meter Profile j
        df_meter_bus_ph1_PQ.columns = ['P', 'Q']
        Phase_A_DF = Phase_A_DF.add(df_meter_bus_ph1_PQ, fill_value=0) 
    
    
    df_meter_bus_ph2 = df_meter_bus[df_meter_bus['Phase1'] == 2]    # Get only Meters with phase 2 out of all Meters of Bus i
    
    for j in df_meter_bus_ph2['Meter_ID']:

        df_meter_bus_ph2_PQ = pd.read_csv('/Users/wania/Desktop/Project/PLX_65019/PQ_csv/P{}.csv'.format(j))      # Reading CSV file of a Meter Profile j
        df_meter_bus_ph2_PQ.columns = ['P', 'Q']
        Phase_B_DF = Phase_B_DF.add(df_meter_bus_ph2_PQ, fill_value=0)

    
    df_meter_bus_ph3 = df_meter_bus[df_meter_bus['Phase1'] == 3]    # Get only Meters with phase 2 out of all Meters of Bus i
    
    for j in df_meter_bus_ph3['Meter_ID']:

        df_meter_bus_ph3_PQ = pd.read_csv('/Users/wania/Desktop/Project/PLX_65019/PQ_csv/P{}.csv'.format(j))      # Reading CSV file of a Meter Profile j
        df_meter_bus_ph3_PQ.columns = ['P', 'Q']
        Phase_C_DF = Phase_C_DF.add(df_meter_bus_ph3_PQ, fill_value=0)

    df_meter_bus_ph4 = df_meter_bus[df_meter_bus['Phase1'] == 4]    # Get only Meters with phase 4 out of all Meters of Bus i
    
    for j in df_meter_bus_ph4['Meter_ID']:

        df_meter_bus_ph4_PQ = pd.read_csv('/Users/wania/Desktop/Project/PLX_65019/PQ_csv/P{}.csv'.format(j))      # Reading CSV file of a Meter Profile j
        df_meter_bus_ph4_PQ.columns = ['P', 'Q']
        Phase_D_DF = Phase_D_DF.add(df_meter_bus_ph4_PQ, fill_value=0)

# Just for testing before concatenating 

#print(Phase_A_DF)
#print(Phase_B_DF)
#print(Phase_C_DF)
#print(Phase_D_DF)

# Concatenating all dataframes before dividing and summing phase 4 yet
test = pd.concat([Phase_A_DF, Phase_B_DF, Phase_C_DF, Phase_D_DF], axis=1)
print(test)


Phase_D_DF = Phase_D_DF/3                                     # Dividing whole Phase 4 dataframe with 3 

Phase_A_DF = Phase_A_DF.add(Phase_D_DF, fill_value=0)        # Adding 1/3 of Phase D to Phase A dataframe
Phase_B_DF = Phase_B_DF.add(Phase_D_DF, fill_value=0)        # Adding 1/3 of Phase D to Phase B dataframe
Phase_C_DF = Phase_C_DF.add(Phase_D_DF, fill_value=0)        # Adding 1/3 of Phase D to Phase C dataframe

Phase_A_DF.columns = ['Pa', 'Qa']                            # Renaming Column names from P and Q to Pa and Qa 
Phase_B_DF.columns = ['Pb', 'Qb']                            # Renaming Column names from P and Q to Pb and Qb 
Phase_C_DF.columns = ['Pc', 'Qc']                            # Renaming Column names from P and Q to Pc and Qc 


# Final dataframe after concatenation

final_df = pd.concat([Phase_A_DF, Phase_B_DF, Phase_C_DF], axis=1)      # axis = 1 shows concatenation along columns
final_df = final_df.round(2)
print(final_df)

# Saving final dataset to csv in working directory
#final_df.to_csv('Final_Dataset.csv')


con.close()
