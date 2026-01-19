import pandas as pd
import os

# --- CONFIGURATION ---
root_folder_path = 'Quebec ER Data' 

all_dataframes = []
skipped_files = [] # To track files that failed to load

print(f"Scanning '{root_folder_path}' for data...")

# 1. TRAVERSE FOLDERS
for root, dirs, files in os.walk(root_folder_path):
    for filename in files:
        if filename.endswith(".csv"):
            file_path = os.path.join(root, filename)
            
            try:
                # 2. READ THE FILE
                current_df = pd.read_csv(file_path, encoding='utf-8')
                
                # 3. STANDARDIZE TIMESTAMP
                if 'Mise_a_jour' in current_df.columns:
                    current_df['Timestamp'] = pd.to_datetime(current_df['Mise_a_jour'])
                else:
                    # If the timestamp column is missing, we can't use this file reliably
                    print(f"[WARNING] Skipping {filename}: 'Mise_a_jour' column missing.")
                    skipped_files.append((filename, "Missing Timestamp Column"))
                    continue
                
                # Add source file for debugging
                current_df['Source_File'] = filename
                
                all_dataframes.append(current_df)
                
            except Exception as e:
                # Log the file that failed and the error message
                print(f"[ERROR] Failed to read {filename}: {e}")
                skipped_files.append((filename, str(e)))

# 4. MERGE EVERYTHING
if all_dataframes:
    print(f"\nMerging {len(all_dataframes)} successfully read files...")
    master_df = pd.concat(all_dataframes, ignore_index=True)
    
    # 5. CLEANUP & SAVE
    master_df.sort_values(by=['Nom_installation', 'Timestamp'], inplace=True)
    output_filename = 'Quebec_ER_Master_Dataset.csv'
    master_df.to_csv(output_filename, index=False)
    
    # --- STATISTICS REPORT ---
    print("\n" + "="*40)
    print("      DATA COLLECTION REPORT")
    print("="*40)
    
    # A. TOTAL HOURS COLLECTED (Total unique timestamps)
    # This counts how many distinct hours (e.g., "Jan 9 13:00") exist in the data
    total_hours_captured = master_df['Timestamp'].nunique()
    print(f"Total Unique Hourly Snapshots: {total_hours_captured}")
    
    # B. TIME SPAN
    min_time = master_df['Timestamp'].min()
    max_time = master_df['Timestamp'].max()
    print(f"Data Starts: {min_time}")
    print(f"Data Ends:   {max_time}")
    duration = max_time - min_time
    print(f"Total Span:  {duration}")

    # C. SKIPPED FILES (WHAT DIDN'T COUNT)
    print("-" * 40)
    print(f"Files Skipped/Failed: {len(skipped_files)}")
    if skipped_files:
        print("List of Failed Files:")
        for name, reason in skipped_files:
            print(f"  - {name} (Reason: {reason})")
    else:
        print("(No files failed to load)")
        
    # D. HOURLY COVERAGE (MISSING HOURS ANALYSIS)
    print("-" * 40)
    print("HOURLY COVERAGE ANALYSIS (Are we missing specific times?)")
    
    master_df['Temp_Hour'] = master_df['Timestamp'].dt.hour
    master_df['Temp_Date'] = master_df['Timestamp'].dt.date
    
    total_unique_days = master_df['Temp_Date'].nunique()
    print(f"Total unique days found: {total_unique_days}")
    
    # Count how many days have data for each hour (0-23)
    hourly_counts = master_df.groupby('Temp_Hour')['Temp_Date'].nunique()
    
    # Identify hours that appear less often than the total number of days
    missing_hours = hourly_counts[hourly_counts < total_unique_days]
    
    if not missing_hours.empty:
        print(f"\n[WARNING] The following hours are missing from some days:")
        for hour, count in missing_hours.items():
            missing_count = total_unique_days - count
            # --- THE FIX IS BELOW: int(hour) ---
            print(f"  - {int(hour):02d}:00 is present on {count} days (Missing on {missing_count} days)")
    else:
        print("\n[PERFECT] Every hour (00:00 to 23:00) is present for every day collected!")

    # Cleanup temp columns
    master_df.drop(columns=['Temp_Hour', 'Temp_Date'], inplace=True)
    print("="*40)

else:
    print("No CSV files were found. Check your folder path.")