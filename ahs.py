import pandas as pd
import numpy as np

def extract_uab_arrival_schedule(file_path):
    # 1. Load and Filter
    df = pd.read_csv(file_path)
    uab = df[df['hospitalName'].str.contains("University of Alberta", case=False, na=False)].copy()
    
    # 2. Clean Timestamps (date.y)
    uab['Timestamp'] = pd.to_datetime(uab['date.y'], format='%d/%m/%Y %H:%M')
    uab = uab.sort_values('Timestamp')
    
    # 3. Aggregate to Hourly Wait Times
    # We use the mean waitTime per hour as our congestion signal
    hourly_data = uab.set_index('Timestamp')['waitTime'].resample('h').mean().fillna(method='ffill')
    
    # 4. Infer Arrivals
    # Since we lack census data, we map the Wait Time to an arrival rate.
    # UofA is a high-volume center (~240 patients/day = ~10/hour).
    # Logic: More wait time = More arrivals relative to capacity.
    mean_wait = hourly_data.mean()
    inferred_arrivals = (hourly_data / mean_wait * 10).round().astype(int)
    
    print(f"Extraction Complete for UofA")
    print(f"Average Wait Time in Data: {mean_wait:.1f} minutes")
    print(f"Max Hourly Arrival (Inferred): {inferred_arrivals.max()}")
    
    return inferred_arrivals.tolist()

if __name__ == "__main__":
    uab_schedule = extract_uab_arrival_schedule('UAB_DATASET.csv')
    print(f"First 24 hours of UofA arrivals: {uab_schedule[:40]}")