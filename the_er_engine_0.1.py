import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# ==========================================
# 1. CONFIGURATION (Based on your Real Data)
# ==========================================
CONFIG = {
    'CAPACITY_BEDS': 33,        # From your data (Royal Vic)
    'SHORT_THRESHOLD': 14.0,    # Reconstructed <33% threshold
    'LONG_THRESHOLD': 32.0,     # Reconstructed >66% threshold
    'CONGESTION_PENALTY': 0.1,  # 10% slowdown when over capacity
    'RANDOM_SEED': 42           # Ensures Policy A vs B are identical comparisons
}

# ==========================================
# 2. THE PATIENT (The Data Object)
# ==========================================
@dataclass
class Patient:
    id: int
    arrival_time: float      # Hour of arrival (e.g., 14.5)
    service_duration_raw: float  # Assigned at birth based on probability
    
    def __post_init__(self):
        self.wait_time = 0.0
        self.completion_time = None
        self.status = "WAITING"  # WAITING, IN_BED, DISCHARGED
        
        # ASSIGN SERVICE CLASS (The "Triage-Lite" Proxy)
        if self.service_duration_raw < CONFIG['SHORT_THRESHOLD']:
            self.service_class = "SHORT"
            self.priority_weight = 3 # Low Priority
        elif self.service_duration_raw > CONFIG['LONG_THRESHOLD']:
            self.service_class = "LONG"
            self.priority_weight = 1 # High Priority
        else:
            self.service_class = "STANDARD"
            self.priority_weight = 2 # Medium Priority

# ==========================================
# 3. THE POLICY MANAGER (The Brain)
# ==========================================
class PolicyManager:
    def __init__(self, mode="BASELINE"):
        self.mode = mode

    def sort_queue(self, queue, current_time):
        """
        Re-orders the waiting list based on the active policy.
        """
        # Helper function to generate sort keys
        def get_sort_key(p):
            
            # --- POLICY A: BASELINE (Triage-Lite) ---
            # Mimics real Quebec prioritization (Sicker/Longer = First)
            if self.mode == "BASELINE":
                return (p.priority_weight, p.arrival_time)
            
            # --- POLICY B: NULL HYPOTHESIS (FCFS) ---
            # Pure First-Come-First-Served
            elif self.mode == "FCFS":
                return (0, p.arrival_time)
            
            # --- POLICY C: GUILLOTINE (>24h Priority) ---
            # If waiting > 24h, super-priority. Else Baseline.
            elif self.mode == "GUILLOTINE":
                is_crisis = (current_time - p.arrival_time) > 24
                # (0 = Crisis, 1 = Normal)
                priority = 0 if is_crisis else 1
                return (priority, p.priority_weight, p.arrival_time)
                
            # --- POLICY D: CONGESTION TRIGGER ---
            # If queue is long (>10 people), use Triage. Else use FCFS.
            elif self.mode == "CONGESTION_TRIGGER":
                is_congested = len(queue) > 10
                if is_congested:
                    return (p.priority_weight, p.arrival_time)
                else:
                    return (0, p.arrival_time)
            
            return (0, p.arrival_time)

        # Apply the sort
        queue.sort(key=get_sort_key)
        return queue

# ==========================================
# 4. DATA LOADER (The Reality Connector)
# ==========================================
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.arrivals_schedule = {} # {Hour: Count}

    def load_royal_vic_data(self):
        print("Loading and filtering Real Data...")
        df = pd.read_csv(self.file_path)
        
        # Filter for Royal Victoria
        vic = df[df['Nom_installation'].str.contains("ROYAL VICTORIA", case=False, na=False)].copy()
        vic['Timestamp'] = pd.to_datetime(vic['Mise_a_jour'])
        vic = vic.sort_values('Timestamp')
        
        # Calculate Arrivals (Reverse engineering from Occupancy)
        # Delta = Occupancy_Now - Occupancy_Prev + Discharges(Approx)
        # Simplified: We use the raw 'Total Patients' change to infer arrival bursts
        # Note: For this engine, we will simplify and use a Replay Generator 
        # based on the average flow to ensure stability.
        
        return vic

# ==========================================
# 5. THE SIMULATION ENGINE (The World)
# ==========================================
class EREngine:
    def __init__(self, policy_name="BASELINE"):
        self.env_time = 0
        self.queue = []
        self.beds = [] # List of active patients in beds
        self.completed_patients = []
        self.policy_manager = PolicyManager(policy_name)
        self.policy_name = policy_name
        
        # Statistics
        self.history_wait_times = []
        
    def spawn_patient(self, p_id):
        # 1. Determine Duration (LogNormal Reconstruction)
        # We draw from a distribution centered on the Real Mean (30h)
        # Sigma=0.8 provides the "Heavy Tail"
        duration = random.lognormvariate(mu=np.log(30), sigma=0.8)
        
        # 2. Create Patient
        p = Patient(p_id, self.env_time, duration)
        self.queue.append(p)
        
    def step(self, hour_index, new_arrivals_count):
        self.env_time = hour_index
        
        # 1. NEW ARRIVALS
        for _ in range(new_arrivals_count):
            p_id = len(self.completed_patients) + len(self.beds) + len(self.queue)
            self.spawn_patient(p_id)
            
        # 2. UPDATE WAIT TIMES
        for p in self.queue:
            p.wait_time += 1.0 # Add 1 hour
            
        # 3. PROCESS DISCHARGES
        # We assume 1 step = 1 hour
        remaining_beds = []
        for p in self.beds:
            p.service_duration_raw -= 1.0
            
            # Soft Limit Logic: If congested, service is slower
            if len(self.beds) > CONFIG['CAPACITY_BEDS']:
                p.service_duration_raw += CONFIG['CONGESTION_PENALTY'] # Add penalty
            
            if p.service_duration_raw <= 0:
                p.status = "DISCHARGED"
                p.completion_time = self.env_time
                self.completed_patients.append(p)
            else:
                remaining_beds.append(p)
        self.beds = remaining_beds
        
        # 4. ADMIT NEW PATIENTS
        # Sort queue based on Policy
        self.policy_manager.sort_queue(self.queue, self.env_time)
        
        # Fill empty beds (Soft Cap: We allow slight overflow to simulate hallway)
        while len(self.beds) < CONFIG['CAPACITY_BEDS'] and self.queue:
            next_patient = self.queue.pop(0)
            next_patient.status = "IN_BED"
            self.beds.append(next_patient)

# ==========================================
# 6. RUNNER & VISUALIZATION
# ==========================================
def run_comparison():
    # Setup - Deterministic Seeds for Fairness
    policies = ["FCFS", "BASELINE", "GUILLOTINE"]
    results = {}
    
    # We simulate 1 week (168 hours)
    # Using a fixed "Crisis" arrival pattern (e.g., 2 patients/hour)
    # In a full version, this comes from DataLoader
    SIM_DURATION = 168 
    ARRIVAL_RATE_PER_HOUR = 2 # Tuned to saturate a 33-bed hospital with 30h stays
    
    for pol in policies:
        print(f"Running Simulation for Policy: {pol}...")
        random.seed(CONFIG['RANDOM_SEED']) # RESET SEED -> Identical Patients
        np.random.seed(CONFIG['RANDOM_SEED'])
        
        sim = EREngine(policy_name=pol)
        
        # Pre-Warm (Hot Start) - Fill the beds first
        for _ in range(35): 
            sim.spawn_patient(999)
            if len(sim.queue) > 0: sim.beds.append(sim.queue.pop(0))
            
        # Run Loop
        for t in range(SIM_DURATION):
            # Poisson Arrival Process
            n_arrivals = np.random.poisson(ARRIVAL_RATE_PER_HOUR)
            sim.step(t, n_arrivals)
            
        # Collect Data
        # We only look at patients who finished or are waiting > 0
        waits = [p.wait_time for p in sim.completed_patients + sim.beds]
        results[pol] = waits

    # --- PLOTTING ---
    plt.figure(figsize=(14, 7))
    
    # Plot 1: Box Plot of Wait Times
    plt.subplot(1, 2, 1)
    data_to_plot = [results[p] for p in policies]
    plt.boxplot(data_to_plot, labels=policies, showfliers=False)
    plt.title("Distribution of Wait Times (Excluding outliers)")
    plt.ylabel("Wait Time (Hours)")
    
    # Plot 2: The Tail (The Guillotine Effect)
    plt.subplot(1, 2, 2)
    for pol in policies:
        # Calculate CDF of tail
        sorted_waits = np.sort(results[pol])
        tail_waits = sorted_waits[sorted_waits > 10] # Look at long waits only
        if len(tail_waits) > 0:
            sns.kdeplot(tail_waits, label=pol, linewidth=2)
            
    plt.axvline(24, color='red', linestyle='--', label="24h Limit")
    plt.title("Tail Risk: Density of Extreme Waits (>10h)")
    plt.xlabel("Wait Time (Hours)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("Final_Policy_Comparison.png")
    print("\nSimulation Complete. Graph saved as 'Final_Policy_Comparison.png'")
    
    # Print Stats
    for pol in policies:
        waits = np.array(results[pol])
        extreme = np.sum(waits > 24)
        print(f"Policy {pol:12} | Avg Wait: {np.mean(waits):.1f}h | Max Wait: {np.max(waits):.1f}h | Patients >24h: {extreme}")

if __name__ == "__main__":
    run_comparison()