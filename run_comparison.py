import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simulation_engine import run_experiment, calibrate_royal_victoria, Patient, ER_Simulation, patient_generator
import simpy

# --- SETUP ---
# We need a custom runner to capture the raw data for graphing
def run_and_capture(file_path, policy_name, days=7):
    capacity, avg_los, arrival_rate = calibrate_royal_victoria(file_path)
    env = simpy.Environment()
    er = ER_Simulation(env, num_beds=capacity, service_rate_mean=avg_los, policy_name=policy_name)
    env.process(patient_generator(env, er, arrival_rate))
    env.run(until=days * 24)
    return [p.wait_time for p in er.patients_processed]

# --- EXECUTION ---
file_name = 'Quebec_ER_Master_Dataset.csv'

print("Running Policy A: First Come First Served (Baseline)...")
waits_fcfs = run_and_capture(file_name, "FCFS")

print("Running Policy B: The Guillotine (>24h Priority)...")
waits_guillotine = run_and_capture(file_name, "Guillotine_24h")

# --- VISUALIZATION (THE WINNING GRAPH) ---
plt.figure(figsize=(12, 6))

# Histogram A (Control)
sns.histplot(waits_fcfs, color="red", alpha=0.5, label="FCFS (Current System)", kde=True, binwidth=5)

# Histogram B (Experiment)
sns.histplot(waits_guillotine, color="blue", alpha=0.5, label="Guillotine Protocol (>24h Priority)", kde=True, binwidth=5)

# The "Crisis Line"
plt.axvline(24, color='black', linestyle='--', linewidth=2, label="24h Target Limit")

plt.title("Impact of 'Guillotine Protocol' on Extreme ER Wait Times (Royal Victoria Simulation)")
plt.xlabel("Wait Time for Bed (Hours)")
plt.ylabel("Number of Patients")
plt.legend()
plt.grid(True, alpha=0.3)

# Save for your report
plt.savefig("Final_Experiment_Results.png")
plt.show()

# --- PRINT STATS ---
def stats(name, data):
    series = pd.Series(data)
    print(f"\n--- {name} ---")
    print(f"Avg Wait: {series.mean():.2f} h")
    print(f"Max Wait: {series.max():.2f} h")
    print(f"Patients > 24h: {len(series[series > 24])}")
    print(f"Reduction in >24h Cases: N/A")

stats("FCFS", waits_fcfs)
stats("Guillotine", waits_guillotine)