import pandas as pd
import numpy as np
import simpy # You may need to install this: pip install simpy
import random
import matplotlib.pyplot as plt

# ==========================================
# PART 1: CALIBRATION (Extracting Real Physics)
# ==========================================
def calibrate_royal_victoria(file_path):
    print("--- CALIBRATING SIMULATION FROM REAL DATA ---")
    df = pd.read_csv(file_path)
    
    # Filter for Royal Victoria
    vic_df = df[df['Nom_installation'].str.contains("ROYAL VICTORIA", case=False, na=False)].copy()
    
    if vic_df.empty:
        raise ValueError("Could not find Royal Victoria data. Check the hospital name.")

    # --- THE FIX: FORCE COLUMNS TO NUMBERS ---
    # This converts "33" (text) to 33 (number) and turns bad data into NaN
    cols_to_clean = ['Nombre_de_civieres_fonctionnelles', 'DMS_sur_civiere', 'Nombre_de_civieres_occupees']
    
    for col in cols_to_clean:
        # errors='coerce' turns unreadable text into NaN (empty), preventing the crash
        vic_df[col] = pd.to_numeric(vic_df[col], errors='coerce')
    
    # Drop rows where critical data is missing after conversion
    vic_df = vic_df.dropna(subset=cols_to_clean)
    # ------------------------------------------

    # 1. Capacity (Number of Stretchers)
    real_capacity = int(vic_df['Nombre_de_civieres_fonctionnelles'].median())
    
    # 2. Service Time (How long do they stay?)
    avg_length_of_stay = vic_df['DMS_sur_civiere'].median()
    
    # 3. Arrival Rate
    # Little's Law: Arrival Rate = Avg Occupied / Avg Stay
    avg_patients_on_stretcher = vic_df['Nombre_de_civieres_occupees'].median()
    
    # Avoid division by zero
    if avg_length_of_stay > 0:
        estimated_arrival_rate = avg_patients_on_stretcher / avg_length_of_stay
    else:
        estimated_arrival_rate = 1.0 # Fallback
    
    print(f"Hospital: Royal Victoria (GLEN)")
    print(f" - Real Capacity (Beds): {real_capacity}")
    print(f" - Avg Length of Stay (Service Time): {avg_length_of_stay:.2f} hours")
    print(f" - Estimated Stretcher Arrival Rate: {estimated_arrival_rate:.2f} patients/hour")
    print("---------------------------------------------")
    
    return real_capacity, avg_length_of_stay, estimated_arrival_rate

# ==========================================
# PART 2: THE SIMULATION ENGINE
# ==========================================

class Patient:
    def __init__(self, p_id, arrival_time):
        self.id = p_id
        self.arrival_time = arrival_time
        self.service_start_time = None
        self.completion_time = None
        self.wait_time = 0
        # You can add 'severity' here later

class ER_Simulation:
    def __init__(self, env, num_beds, service_rate_mean, policy_name="FCFS"):
        self.env = env
        self.beds = simpy.Resource(env, num_beds)
        self.service_rate_mean = service_rate_mean
        self.policy_name = policy_name
        
        self.queue = [] # Our waiting room
        self.patients_processed = []
        self.logs = []

    def arrive(self, patient):
        # Log arrival
        self.queue.append(patient)
        self.logs.append((self.env.now, 'Arrival', patient.id, len(self.queue)))
        
        # TRIGGER POLICY: Sort the queue based on the rule
        self.apply_policy()
        
        # Request a bed
        with self.beds.request() as request:
            yield request # Wait until a bed is free
            
            # Bed found!
            self.queue.remove(patient)
            patient.wait_time = self.env.now - patient.arrival_time
            patient.service_start_time = self.env.now
            
            self.logs.append((self.env.now, 'Admit', patient.id, len(self.queue)))
            
            # Simulate Treatment (Service Time)
            # We use an Exponential distribution (standard for service times)
            service_time = random.expovariate(1.0 / self.service_rate_mean)
            yield self.env.timeout(service_time)
            
            # Discharge
            patient.completion_time = self.env.now
            self.patients_processed.append(patient)

    def apply_policy(self):
        """
        THIS IS WHERE YOUR SCIENCE HAPPENS.
        Re-order self.queue based on your logic.
        """
        if self.policy_name == "FCFS":
            # Sort by arrival time (Ascending) - Standard
            self.queue.sort(key=lambda p: p.arrival_time)
            
        elif self.policy_name == "Guillotine_24h":
            # If wait > 24h, move to front. Else FCFS.
            # (In sim time, 24h = 24 units)
            current_time = self.env.now
            self.queue.sort(key=lambda p: (0 if (current_time - p.arrival_time) > 24 else 1, p.arrival_time))

# ==========================================
# PART 3: THE GENERATOR
# ==========================================
def patient_generator(env, er_sim, arrival_rate):
    p_id = 0
    while True:
        # Generate next arrival time (Poisson process)
        interarrival_time = random.expovariate(arrival_rate)
        yield env.timeout(interarrival_time)
        
        p_id += 1
        p = Patient(p_id, env.now)
        env.process(er_sim.arrive(p))

# ==========================================
# PART 4: RUNNER & EVALUATION
# ==========================================
def run_experiment(file_path, simulation_days=7, policy="FCFS"):
    # 1. Get Real Parameters
    capacity, avg_los, arrival_rate = calibrate_royal_victoria(file_path)
    
    # 2. Setup SimPy Environment
    env = simpy.Environment()
    er = ER_Simulation(env, num_beds=capacity, service_rate_mean=avg_los, policy_name=policy)
    
    # 3. Start Generator
    env.process(patient_generator(env, er, arrival_rate))
    
    # 4. Run
    print(f"Starting Simulation ({policy})...")
    env.run(until=simulation_days * 24) # Run for X days
    
    # 5. Analyze Results
    df_results = pd.DataFrame([vars(p) for p in er.patients_processed])
    
    if not df_results.empty:
        avg_wait = df_results['wait_time'].mean()
        max_wait = df_results['wait_time'].max()
        long_waits = len(df_results[df_results['wait_time'] > 24])
        
        print("\n--- SIMULATION RESULTS ---")
        print(f"Policy: {policy}")
        print(f"Total Patients Treated: {len(df_results)}")
        print(f"Average Wait Time: {avg_wait:.2f} hours")
        print(f"Max Wait Time: {max_wait:.2f} hours")
        print(f"Patients waited > 24h: {long_waits}")
        
        return avg_wait, max_wait
    else:
        print("No patients finished treatment.")
        return 0, 0

# --- EXECUTION BLOCK ---
# Replace with your actual file name
if __name__ == "__main__":
    # Ensure you have 'simpy' installed: pip install simpy
    file_name = 'Quebec_ER_Master_Dataset.csv' 
    
    # Run Baseline (Control)
    run_experiment(file_name, policy="FCFS")
    
    # Run Experiment (Test)
    # run_experiment(file_name, policy="Guillotine_24h")