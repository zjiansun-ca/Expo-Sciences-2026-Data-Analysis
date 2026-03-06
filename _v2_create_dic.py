import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_arrival_schedule(file_path, hospital_name="ROYAL VICTORIA"):
    print(f"--- EXTRACTING REAL ARRIVAL SCHEDULE FOR {hospital_name} ---")
    df = pd.read_csv(file_path)
    
    # Filter and Sort
    vic = df[df['Nom_installation'].str.contains(hospital_name, case=False, na=False)].copy()
    
    # --- THE FIX IS HERE ---
    # format='mixed' handles variations in the timestamp strings
    # errors='coerce' turns completely broken strings into NaT (Not a Time)
    vic['Timestamp'] = pd.to_datetime(vic['Mise_a_jour'], format='mixed', errors='coerce')
    
    # Drop any rows where the timestamp was completely unreadable
    vic = vic.dropna(subset=['Timestamp']) 
    vic = vic.sort_values('Timestamp')
    # -----------------------
    
    # Remove duplicates if any (same hour reported twice)
    vic = vic.drop_duplicates(subset=['Timestamp'], keep='last')
    
    # Calculate Arrivals logic
    vic['Occupied'] = pd.to_numeric(vic['Nombre_de_civieres_occupees'], errors='coerce').fillna(0)
    vic['Prev_Occupied'] = vic['Occupied'].shift(1).fillna(vic['Occupied'])
    
    # Estimate Discharges (Little's Law estimation per step)
    vic['Est_Discharges'] = vic['Occupied'] / 30.0 
    
    # Calculate Raw Arrivals
    vic['Arrivals_Raw'] = (vic['Occupied'] - vic['Prev_Occupied']) + vic['Est_Discharges']
    
    # Clean up negatives and noise (smoothing)
    vic['Arrivals_Clean'] = vic['Arrivals_Raw'].apply(lambda x: max(0, int(round(x))))
    
    # Save to a simple list/dictionary for the engine
    arrival_schedule = vic['Arrivals_Clean'].tolist()
    
    print(f"Extracted {len(arrival_schedule)} hours of data.")
    print(f"Total Estimated Arrivals: {sum(arrival_schedule)}")
    print(f"Average Arrivals/Hour: {np.mean(arrival_schedule):.2f}")
    
    # Plot to verify it looks like a daily cycle
    plt.figure(figsize=(12, 4))
    plt.plot(arrival_schedule[:168]) # First week
    plt.title("Estimated Hourly Arrivals (First 7 Days)")
    plt.xlabel("Hour")
    plt.ylabel("New Patients")
    plt.savefig("arrival_pattern_check.png")
    print("Saved 'arrival_pattern_check.png'")
    
    return arrival_schedule

if __name__ == "__main__":
    schedule = extract_arrival_schedule('Quebec_ER_Master_Dataset.csv')
    print(f"\nCOPY THIS LIST START: {schedule[:40]} ...")