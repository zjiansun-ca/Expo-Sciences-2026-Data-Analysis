import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_arrival_schedule_improved(file_path, hospital_name="ROYAL VICTORIA"):
    df = pd.read_csv(file_path)
    
    # 1. Filter and Clean
    vic = df[df['Nom_installation'].str.contains(hospital_name, case=False, na=False)].copy()
    vic['Timestamp'] = pd.to_datetime(vic['Mise_a_jour'], format='mixed', errors='coerce')
    vic = vic.dropna(subset=['Timestamp']).sort_values('Timestamp')
    vic = vic.drop_duplicates(subset=['Timestamp'], keep='last')
    
    # 2. Convert to Numeric
    metrics = ['Nombre_total_de_patients_presents_a_lurgence', 'Nombre_de_civieres_occupees', 
               'DMS_sur_civiere', 'DMS_ambulatoire']
    for m in metrics:
        vic[m] = pd.to_numeric(vic[m], errors='coerce').fillna(0)
    
    # 3. Separate Populations
    vic['N_stretcher'] = vic['Nombre_de_civieres_occupees']
    vic['N_ambulatory'] = (vic['Nombre_total_de_patients_presents_a_lurgence'] - vic['N_stretcher']).clip(lower=0)
    
    # 4. Calculate Discharges using dynamic DMS (Length of Stay)
    # If DMS is 0, we use a conservative fallback (24h for stretcher, 4h for walking)
    vic['Rate_Stretcher'] = 1 / vic['DMS_sur_civiere'].replace(0, 24)
    vic['Rate_Ambulatory'] = 1 / vic['DMS_ambulatoire'].replace(0, 4)
    
    vic['Est_Discharges'] = (vic['N_stretcher'].shift(1) * vic['Rate_Stretcher'].shift(1)) + \
                            (vic['N_ambulatory'].shift(1) * vic['Rate_Ambulatory'].shift(1))
    
    # 5. The Arrival Formula: (Net Change in Census) + (Outflow)
    vic['Census_Delta'] = vic['Nombre_total_de_patients_presents_a_lurgence'].diff()
    vic['Arrivals_Raw'] = vic['Census_Delta'] + vic['Est_Discharges']
    
    # Clean noise (round to nearest whole patient, no negatives)
    vic['Arrivals_Final'] = vic['Arrivals_Raw'].fillna(0).apply(lambda x: int(round(max(0, x))))
    
    # Export and Stats
    print(f"Average Arrivals/Hour: {vic['Arrivals_Final'].mean():.2f}")
    print(f"Estimated Peak Hourly Arrival: {vic['Arrivals_Final'].max()}")
    
    return vic['Arrivals_Final'].tolist()

if __name__ == "__main__":
    schedule = extract_arrival_schedule_improved('Quebec_ER_Master_Dataset.csv')
    print(f"First 24 hours of arrivals: {schedule[:40]}")