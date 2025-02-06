#Importing packages/libraries

import pandas as pd
import sqlite3
from pandasql import sqldf
import numpy as np

# Read sqlite query results into a pandas DataFrame
con = sqlite3.connect("/Users/wania/Desktop/Project/PLX_65019/net.db")

# Read Bus IDs into a dataframe except 1st bus
bus_ids = pd.read_sql_query("SELECT Bus.ID from Bus", con).drop(index=0) 
#print(bus_ids)


voltage_df = pd.DataFrame()
for i in bus_ids['ID']:

    #Reading Voltages file for Bus ID i. 
    bus_volts = pd.read_csv('/Users/wania/Desktop/Project/PLX_65019/out/TestNet_Mon_b{}_1.csv'.format(i)) 

    bus_volts.columns = bus_volts.columns.str.strip()               # Else gives error

    # Calculating - sqrt(x*x + y*y)
    v1_mag = np.sqrt(bus_volts['V1.re']**2 + bus_volts['V1.im']**2)
    v2_mag = np.sqrt(bus_volts['V2.re']**2 + bus_volts['V2.im']**2)
    v3_mag = np.sqrt(bus_volts['V3.re']**2 + bus_volts['V3.im']**2)
    v4_mag = np.sqrt(bus_volts['V4.re']**2 + bus_volts['V4.im']**2)

    # Calculating - sqrt(3) * sqrt(x*x + y*y) / 400.0        
    voltage_df[['Hour','t(sec)']] = bus_volts[['hour','t(sec)']]
    voltage_df['V1_Magnitude'] = ((np.sqrt(3)) * (v1_mag/ 400.0))
    voltage_df['V2_Magnitude'] = ((np.sqrt(3)) * (v2_mag/ 400.0))
    voltage_df['V3_Magnitude'] = ((np.sqrt(3)) * (v3_mag/ 400.0))
    voltage_df['V4_Magnitude'] = ((np.sqrt(3)) * (v4_mag/ 400.0))
    
    # Save dataframe to a new file 
    voltage_df.to_csv('/Users/wania/Desktop/Project/PLX_65019/output_bus_voltage/Bus_{}_Volts.csv'.format(i))
    
