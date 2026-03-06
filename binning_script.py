import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_distribution(file_path, hospital_name="ROYAL VICTORIA"):
    print(f"--- ANALYZING DISTRIBUTION FOR: {hospital_name} ---")
    
    # 1. Load & Clean
    df = pd.read_csv(file_path)
    
    # Filter for the specific hospital
    # We use 'contains' to be safe with naming variations
    target_df = df[df['Nom_installation'].str.contains(hospital_name, case=False, na=False)].copy()
    
    if target_df.empty:
        print(f"ERROR: No data found for {hospital_name}.")
        return

    # Convert DMS to numeric (coercing errors to NaN)
    target_df['DMS_ambulatoire'] = pd.to_numeric(target_df['DMS_ambulatoire'], errors='coerce')
    clean_data = target_df.dropna(subset=['DMS_ambulatoire'])
    
    # 2. Calculate Statistics
    # We use the 33rd and 66th percentiles to define our 3 classes
    percentiles = clean_data['DMS_ambulatoire'].quantile([0.33, 0.66]).values
    t_short = percentiles[0]
    t_long = percentiles[1]
    
    mean_val = clean_data['DMS_ambulatoire'].mean()
    max_val = clean_data['DMS_ambulatoire'].max()
    
    print(f"\n[RESULTS]")
    print(f"Total Hourly Snapshots: {len(clean_data)}")
    print(f"Mean Length of Stay:    {mean_val:.2f} hours")
    print(f"Max Length of Stay:     {max_val:.2f} hours")
    print("-" * 30)
    print(f"THRESHOLD: Short Stay (<33%): < {t_short:.2f} hours")
    print(f"THRESHOLD: Long Stay (>66%):  > {t_long:.2f} hours")
    print("-" * 30)

    # 3. Visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(clean_data['DMS_ambulatoire'], bins=30, kde=True, color='skyblue')
    plt.axvline(t_short, color='green', linestyle='--', label=f'Short/Std Cutoff ({t_short:.1f}h)')
    plt.axvline(t_long, color='red', linestyle='--', label=f'Std/Long Cutoff ({t_long:.1f}h)')
    plt.title(f'Distribution of ER Service Times ({hospital_name})')
    plt.xlabel('Average Length of Stay (Hours)')
    plt.ylabel('Frequency (Count of Hours)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save graph
    plt.savefig('distribution_check.png')
    print("Graph saved as 'distribution_check.png'")

# --- RUN IT ---
if __name__ == "__main__":
    # Update this filename if needed
    analyze_distribution('Quebec_ER_Master_Dataset.csv')