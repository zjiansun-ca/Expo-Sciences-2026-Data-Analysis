import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. LOAD
#df = pd.read_csv('Quebec_ER_Master_Dataset.csv')
df = pd.read_csv('Quebec_ER_Master_Dataset.csv')

# 2. CLEANING
# Remove the "Total régional" rows to avoid double counting
df = df[df['Nom_etablissement'] != 'Total régional']

# Convert columns to numeric (just in case they loaded as text)
cols_to_fix = ['Nombre_de_civieres_occupees', 'Nombre_de_civieres_fonctionnelles', 'DMS_ambulatoire']
for col in cols_to_fix:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. FEATURE ENGINEERING (Create the KPIs)
# Avoid division by zero
df = df[df['Nombre_de_civieres_fonctionnelles'] > 0] 
df['Occupancy_Rate'] = (df['Nombre_de_civieres_occupees'] / df['Nombre_de_civieres_fonctionnelles']) * 100

# --- ANALYSIS A: REGIONAL DISPARITIES ---
# Group by Region and calculate the mean Occupancy and Wait Time
regional_stats = df.groupby('Region')[['Occupancy_Rate', 'DMS_ambulatoire']].mean().sort_values('Occupancy_Rate', ascending=False)

print("Top 5 Most Overcrowded Regions:")
print(regional_stats.head())

# Plotting the Regional Comparison
plt.figure(figsize=(12, 6))
sns.barplot(x=regional_stats.index, y=regional_stats['Occupancy_Rate'], palette='Reds_r')
plt.axhline(100, color='black', linestyle='--', label='100% Capacity')
plt.xticks(rotation=45, ha='right')
plt.title('Average ER Occupancy Rate by Region (Last 9 Days)')
plt.ylabel('Occupancy Rate (%)')
plt.legend()
plt.tight_layout()
plt.show()

# --- ANALYSIS B: THE PULSE (TIME SERIES) ---
# We need to see the "average" day. 
# We extract the "Hour" from the timestamp to see the daily cycle.
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour

# Group by Hour to see the "Average Day" pattern across the whole province
hourly_pulse = df.groupby('Hour')['Occupancy_Rate'].mean()

plt.figure(figsize=(10, 5))
sns.lineplot(x=hourly_pulse.index, y=hourly_pulse.values, linewidth=3, color='darkblue')
plt.title('The "Heartbeat" of Quebec ERs: Average Occupancy by Hour of Day')
plt.xlabel('Hour of Day (0-23)')
plt.ylabel('Average Occupancy Rate (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(0, 24))
plt.show()