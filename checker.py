import pandas as pd

df = pd.read_csv('Quebec_ER_Master_Dataset.csv')

# Check how many unique time points you have per hospital
# (This tells you if your "missing hours" are bad or manageable)
print(df.groupby('Nom_installation')['Timestamp'].nunique())