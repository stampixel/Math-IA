# Transforming qualitative data to quantitative data

import pandas as pd

df = pd.read_csv('cell_data.csv')
print(df)
for i, row in df.iterrows():
    # print(i, row)
    if df.at[i, 'Cluster'] == "r":
        df.at[i, 'Cluster'] = 0
    elif df.at[i, 'Cluster'] == "g":
        df.at[i, 'Cluster'] = 1
    elif df.at[i, 'Cluster'] == "b":
        df.at[i, 'Cluster'] = 2
    df.at[i, 'X-axis'] = round(df.at[i, 'X-axis'])
    df.at[i, 'Y-axis'] = round(df.at[i, 'Y-axis'])

print(df.Cluster)
df.to_csv('data.csv', index=False)
print(df)
