import pandas as pd
import matplotlib.pyplot as plt



dataframe = pd.read_csv('/Users/wania/Desktop/Project/PLX_65019/Aggregated_Data/Bus_41627_PQ_Volts.csv', index_col=0)
#print(dataframe)

x = dataframe[['Pa', 'Qa', 'Pb', 'Qb','Pc','Qc']]  

y = dataframe[['Hour']] 
  

print("Skewness of training features:\n", dataframe[['Pa', 'Qa', 'Pb', 'Qb','Pc','Qc']].skew())

print("Skewness of target features:\n ",dataframe[['V1_Magnitude', 'V2_Magnitude','V3_Magnitude']].skew())

#dataframe[['Pa', 'Qa', 'Pb', 'Qb','Pc','Qc','V1_Magnitude', 'V2_Magnitude','V3_Magnitude']].hist()


fig, axes = plt.subplots(3,3)
axes[0, 0].hist(dataframe[['Pa']], label=['Pa'])
axes[0, 1].hist(dataframe[['Pb']], label=['Pb'])
axes[0, 2].hist(dataframe[['Pc']], label=['Pc'])
axes[1, 0].hist(dataframe[['Qa']], label=['Qa'])
axes[1, 1].hist(dataframe[['Qb']], label=['Qb'])
axes[1, 2].hist(dataframe[['Qc']], label=['Qc'])
axes[2, 0].hist(dataframe[['V1_Magnitude']], label=['V1_Magnitude'])
axes[2, 1].hist(dataframe[['V2_Magnitude']], label=['V2_Magnitude'])
axes[2, 2].hist(dataframe[['V3_Magnitude']], label=['V3_Magnitude'])

# Customize the plot
fig.suptitle('Histograms of Training and Target Features')
axes[0, 0].set_xlabel('Pa Values')
axes[0, 0].set_ylabel('Frequency')
axes[0, 1].set_xlabel('Pb Values')
axes[0, 1].set_ylabel('Frequency')
axes[0, 2].set_xlabel('Pc Values')
axes[0, 2].set_ylabel('Frequency')

axes[1, 0].set_xlabel('Qa Values')
axes[1, 0].set_ylabel('Frequency')
axes[1, 1].set_xlabel('Qb Values')
axes[1, 1].set_ylabel('Frequency')
axes[1, 2].set_xlabel('Qc Values')
axes[1, 2].set_ylabel('Frequency')

axes[2, 0].set_xlabel('V1 Magnitude')
axes[2, 0].set_ylabel('Frequency')
axes[2, 1].set_xlabel('V2 Magnitude')
axes[2, 1].set_ylabel('Frequency')
axes[2, 2].set_xlabel('V3 Magnitude')
axes[2, 2].set_ylabel('Frequency')

#plt.title('Histograms')
plt.legend()
plt.show()