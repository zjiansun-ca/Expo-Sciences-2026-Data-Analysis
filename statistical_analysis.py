import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. SETUP
# Load your master dataset
df = pd.read_csv('Quebec_ER_Master_Dataset.csv')

# Filter out regional totals
df = df[df['Nom_etablissement'] != 'Total régional']

# --- THE FIX IS HERE ---
# 2. CLEANING DATA TYPES
# We force these columns to be numbers. 'coerce' turns bad data (like text) into NaN (empty)
cols_to_fix = ['Nombre_de_civieres_fonctionnelles', 'Nombre_de_civieres_occupees', 'DMS_ambulatoire']

for col in cols_to_fix:
    # This line fixes the "TypeError" you saw
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where critical data is missing (NaN) after conversion
df = df.dropna(subset=cols_to_fix)
# -----------------------

# Now this line will work because the column is numbers
df = df[df['Nombre_de_civieres_fonctionnelles'] > 0] 

# 3. CREATE VARIABLES
# Occupancy Rate (%)
df['Occupancy_Rate'] = (df['Nombre_de_civieres_occupees'] / df['Nombre_de_civieres_fonctionnelles']) * 100

# We focus on "Ambulatory Wait" (DMS_ambulatoire)
target_variable = 'DMS_ambulatoire' 

# 4. STATISTICAL BINNING (The "Threshold" Analysis)
bins = [0, 60, 80, 100, 120, 150, 500]
labels = ['<60%', '60-80%', '80-100%', '100-120%', '120-150%', '>150%']
df['Occupancy_Bin'] = pd.cut(df['Occupancy_Rate'], bins=bins, labels=labels)

# 5. VISUALIZATION 1: The "Hockey Stick" Scatter Plot
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Occupancy_Rate', y=target_variable, alpha=0.1, color='gray', s=10)

# Add a "Trend Line" using a rolling median
df_sorted = df.sort_values('Occupancy_Rate')
rolling_median = df_sorted[target_variable].rolling(window=500, center=True).median()
plt.plot(df_sorted['Occupancy_Rate'], rolling_median, color='red', linewidth=3, label='Median Trend Line')

plt.xlim(0, 200) 
plt.ylim(0, df[target_variable].quantile(0.99)) 
plt.title(f'The Tipping Point: Occupancy vs. {target_variable}')
plt.xlabel('Occupancy Rate (%)')
plt.ylabel('Average Wait Time (Hours)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('graph_tipping_point.png') # Saves the graph
plt.show()

# 6. VISUALIZATION 2: The Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Occupancy_Bin', y=target_variable, palette="Reds")
plt.title(f'Wait Time Distribution by Occupancy Level')
plt.xlabel('Occupancy Range')
plt.ylabel('Wait Time (Hours)')
plt.grid(axis='y', alpha=0.3)
plt.savefig('graph_boxplot.png') # Saves the graph
plt.show()

# 7. CALCULATE THE "JUMP"
summary = df.groupby('Occupancy_Bin', observed=True)[target_variable].median()
print("Median Wait Times by Occupancy Level:")
print(summary)