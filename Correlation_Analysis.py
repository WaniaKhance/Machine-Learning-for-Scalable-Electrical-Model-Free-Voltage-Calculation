import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dataframe = pd.read_csv('/Users/wania/Desktop/Project/PLX_65019/Aggregated_Data/Bus_41627_PQ_Volts.csv', index_col=0)
df = dataframe[['Pa', 'Qa', 'Pb', 'Qb','Pc','Qc','V1_Magnitude', 'V2_Magnitude','V3_Magnitude']]
print("\n\n\n                       Pearson's Correlation Analysis between Training and Target Features\n\n",df.corr(method ='pearson'),"\n\n\n")

# fig, ax = plt.subplots(figsize=(10, 6))

# sns.heatmap(df.corr(), ax=ax, annot=True)


fig, ax = plt.subplots(1,3, figsize=(18, 8))

corr1 = df.corr('spearman')[['V1_Magnitude']].sort_values(by='V1_Magnitude', ascending=False)
corr2 = df.corr('spearman')[['V2_Magnitude']].sort_values(by='V2_Magnitude', ascending=False)
corr3 = df.corr('spearman')[['V3_Magnitude']].sort_values(by='V3_Magnitude', ascending=False)


sns.heatmap(corr1, ax=ax[0], annot=True)
sns.heatmap(corr2, ax=ax[1], annot=True)
sns.heatmap(corr3, ax=ax[2], annot=True)


plt.suptitle("Spearman's Correlation Analysis of Training and Target Features")
plt.show()