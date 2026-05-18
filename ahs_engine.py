import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# ==========================
# 1. SCIENTIFIC CONFIGURATION
# ==========================
CONFIG = {
    'NOMINAL_CAPACITY': 75,      # UofA is significantly larger than Victoria
    'MAX_OVERCROWD': 1.3,       # High tolerance for surge
    'CONGESTION_PENALTY': 0.25,  # Efficiency drops more in large centers
    'TRIAGE_ERROR_RATE': 0.08, 
    'SHORT_THRESHOLD': 12.0,   
    'LONG_THRESHOLD': 36.0,     
    # New policy thresholds
    'ESCALATION_L2_THRESHOLD': 8,   # L2 escalates after 8h wait
    'ESCALATION_L3_THRESHOLD': 15,  # L3 escalates after 15h wait
}

@dataclass
class Patient:
    id: int
    arrival_time: int
    true_service_duration: float 
    perceived_severity: int      
    
    def __post_init__(self):
        self.wait_time = 0
        self.status = "WAITING" 
        self.time_in_bed = 0

class EREngineV3:
    def __init__(self, arrival_schedule, policy_mode="BASELINE"):
        self.schedule = arrival_schedule
        self.policy_mode = policy_mode
        self.env_time = 0
        
        self.queue = []
        self.beds = []
        self.completed = []
        self.history_queue_length = [] 
        
    def spawn_patients(self, count, offset_time=None):
        for _ in range(int(count)):
            p_id = len(self.queue) + len(self.beds) + len(self.completed)
            
            # Service Duration Distribution (Adjusted for UofA Complexity)
            # High-acuity trauma centers have longer tails in service time
            true_duration = random.lognormvariate(mu=np.log(12), sigma=0.7)
            
            # Probability Distribution for UofA (High Acuity)
            # 15% Level 1 (Severe), 45% Level 2 (Standard), 40% Level 3 (Minor)
            rand_val = random.random()
            if rand_val < 0.05: base_severity = 1
            elif rand_val < 0.80: base_severity = 2
            else: base_severity = 3
                
            arr_time = self.env_time if offset_time is None else offset_time
            p = Patient(p_id, arr_time, true_duration, base_severity)
            self.queue.append(p)

    def sort_queue(self):
        if self.policy_mode == "FCFS":
            self.queue.sort(key=lambda p: -p.wait_time)
            
        elif self.policy_mode == "BASELINE":
            self.queue.sort(key=lambda p: (p.perceived_severity, -p.wait_time))
            
        elif self.policy_mode == "GUILLOTINE":
            self.queue.sort(key=lambda p: (
                0 if p.wait_time > 10 else p.perceived_severity, 
                -p.wait_time
            ))
            
        elif self.policy_mode == "FAST_TRACK":
            def fast_track_priority(p):
                if p.perceived_severity == 1: return 1
                elif p.perceived_severity == 2 and p.wait_time > 10: return 1.5 
                elif p.perceived_severity == 3: return 2
                else: return 3 
                
            self.queue.sort(key=lambda p: (fast_track_priority(p), -p.wait_time))
        
        elif self.policy_mode == "WEIGHTED_FCFS":
            def max_min_priority(p):
                if p.perceived_severity == 1:
                    return (0, -p.wait_time)   # L1: severity-first as always
                else:
                    return (1, -p.wait_time)   # L2+L3: pure wait time, no distinction
            
            self.queue.sort(key=max_min_priority)

    def step(self):
        if self.env_time < len(self.schedule):
            new_count = self.schedule[self.env_time]
        else:
            new_count = 0 
            
        self.spawn_patients(new_count)
        
        for p in self.queue:
            p.wait_time += 1
            
        current_occupancy = len(self.beds) / CONFIG['NOMINAL_CAPACITY']
        efficiency = 1.0
        if current_occupancy > 1.0:
            efficiency -= CONFIG['CONGESTION_PENALTY']
            
        remaining_beds = []
        for p in self.beds:
            p.time_in_bed += 1
            progress = 1.0 * efficiency
            
            if p.time_in_bed * progress >= p.true_service_duration:
                p.status = "DISCHARGED"
                self.completed.append(p)
            else:
                remaining_beds.append(p)
        self.beds = remaining_beds
        
        self.sort_queue()
        real_limit = int(CONFIG['NOMINAL_CAPACITY'] * CONFIG['MAX_OVERCROWD'])
        
        while self.queue and len(self.beds) < real_limit:
            p = self.queue.pop(0)
            p.status = "IN_BED"
            self.beds.append(p)
            
        self.history_queue_length.append(len(self.queue))
        self.env_time += 1

# ==========================
# 3. RUNNER & 6-GRAPH ANALYTICS
# ==========================
def run_v3_experiment(arrival_data):
    policies = ["FCFS", "BASELINE", "GUILLOTINE", "FAST_TRACK", "WEIGHTED_FCFS"]
    colors = ['#95a5a6', '#4f83d1', '#f39c12', '#2ecc71', '#9b59b6']  # +Purple
    
    results_map = {}
    queue_history = {}
    severity_waits = {pol: {1: [], 2: [], 3: []} for pol in policies}
    
    print("\n" + "="*50)
    print("ER SIMULATION ANALYTICS DASHBOARD")
    print("="*50)
    
    for pol in policies:
        random.seed(42)
        np.random.seed(42)
        
        engine = EREngineV3(arrival_data, policy_mode=pol)
        
        for _ in range(65):
            past_time = -int(random.uniform(0, 48))
            engine.spawn_patients(1, offset_time=past_time)
        
        engine.sort_queue()
        real_limit = int(CONFIG['NOMINAL_CAPACITY'] * CONFIG['MAX_OVERCROWD'])
        while engine.queue and len(engine.beds) < real_limit:
            p = engine.queue.pop(0)
            p.status = "IN_BED"
            engine.beds.append(p)
        
        for _ in range(len(arrival_data)):
            engine.step()
            
        all_patients = engine.completed + engine.beds + engine.queue
        waits = [p.wait_time for p in all_patients]
        results_map[pol] = waits
        queue_history[pol] = engine.history_queue_length
        
        for p in all_patients:
            severity_waits[pol][p.perceived_severity].append(p.wait_time)
        
        avg = np.mean(waits)
        median = np.median(waits)
        max_w = np.max(waits)
        bad_w = sum(1 for w in waits if w > 24)
        crit_avg = np.mean(severity_waits[pol][1]) if severity_waits[pol][1] else 0
        
        print(f"\nPolicy: {pol}")
        print(f"  Overall -> Median: {median:.1f}h | Avg: {avg:.1f}h | Max: {max_w:.1f}h | >24h: {bad_w}")
        print(f"  Critical-> Avg Wait for Severe Patients (Level 1): {crit_avg:.1f}h")

    # GRAPH GENERATION (6 GRAPHS)
    
    # Graph 1: The Boxplot
    plt.figure(figsize=(11, 6))
    bplot = plt.boxplot([results_map[p] for p in policies], tick_labels=policies, showfliers=False, patch_artist=True)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.title("Graph 1: Spread of Wait Times (Middle 75%)")
    plt.ylabel("Wait Time (Hours)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("AHS_Graph1_Boxplot.png")
    
    # Graph 2: Waiting Room Size Over Time
    plt.figure(figsize=(11, 6))
    for i, pol in enumerate(policies):
        plt.plot(queue_history[pol], label=pol, linewidth=2, color=colors[i])
    plt.title("Graph 2: Waiting Room Congestion Over Time")
    plt.xlabel("Hours Since Simulation Started")
    plt.ylabel("Patients in Waiting Room")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("AHS_Graph2_Queue_Size.png")
    
    # Graph 3: Wait Time by Severity Level
    plt.figure(figsize=(12, 6))
    bar_width = 0.2
    index = np.arange(len(policies))
    
    avg_sev1 = [np.mean(severity_waits[p][1]) for p in policies]
    avg_sev2 = [np.mean(severity_waits[p][2]) for p in policies]
    avg_sev3 = [np.mean(severity_waits[p][3]) for p in policies]
    
    plt.bar(index, avg_sev1, bar_width, label='Severe (Level 1)', color='#c0392b', alpha=0.9)
    plt.bar(index + bar_width, avg_sev2, bar_width, label='Standard (Level 2)', color='#e67e22', alpha=0.9)
    plt.bar(index + 2*bar_width, avg_sev3, bar_width, label='Minor (Level 3)', color='#f1c40f', alpha=0.9)
    
    plt.xlabel('Triage Policy')
    plt.ylabel('Average Wait Time (Hours)')
    plt.title('Graph 3: Ethical Trade-offs (Who Waits Longer?)')
    plt.xticks(index + bar_width, policies, rotation=10)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("AHS_Graph3_Severity_Tradeoff.png")

    # Graph 4: CDF
    plt.figure(figsize=(11, 6))
    for i, pol in enumerate(policies):
        sorted_waits = np.sort(results_map[pol])
        yvals = np.arange(1, len(sorted_waits)+1) / len(sorted_waits) * 100
        plt.plot(sorted_waits, yvals, label=pol, linewidth=2.5, color=colors[i])
    
    plt.axvline(x=24, color='black', linestyle='--', alpha=0.6, label='24h Danger Zone')
    plt.title("Graph 4: Cumulative Success (% of Patients Seen Within X Hours)")
    plt.xlabel("Wait Time (Hours)")
    plt.ylabel("Percentage of Patients Seen (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 50) 
    plt.tight_layout()
    plt.savefig("AHS_Graph4_Cumulative_Success.png")

    # Graph 5: >24h Crisis Bar Chart
    plt.figure(figsize=(9, 6))
    bad_counts = [sum(1 for w in results_map[p] if w > 24) for p in policies]
    bars = plt.bar(policies, bad_counts, color=colors, alpha=0.9)
    plt.title("Graph 5: The 24-Hour Crisis (Total Patients Stranded)")
    plt.ylabel("Number of Patients Waiting > 24 Hours")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("AHS_Graph5_Crisis_Count.png")

    # Graph 6: KDE Density
    plt.figure(figsize=(11, 6))
    for i, pol in enumerate(policies):
        sns.kdeplot(results_map[pol], label=pol, color=colors[i], linewidth=2.5, fill=True, alpha=0.1)
    plt.title("Graph 6: Density Distribution of Wait Times")
    plt.xlabel("Wait Time (Hours)")
    plt.ylabel("Density of Patients")
    plt.xlim(0, 50)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("AHS_Graph6_Density_Curve.png")

    print("\n" + "="*50)
    print("SUCCESS: 6 High-Quality Graphs Generated!")
    print("Check your folder for Graph1 through Graph6.")

# USAGE
real_arrivals_list = [8, 8, 9, 11, 9, 7, 8, 9, 12, 14, 17, 13, 12, 11, 10, 12, 11, 6, 6, 10, 5, 7, 10, 11, 9, 9, 12, 13, 16, 14, 13, 17, 18, 21, 18, 16, 19, 20, 22, 20]
run_v3_experiment(real_arrivals_list)