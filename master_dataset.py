import pandas as pd
import os

# --- CONFIGURATION ---
# Replace this with the exact path to your main folder
# If the folder is in the same place as your script, just use 'Quebec ER Data'
root_folder_path = 'Quebec ER Data' 

all_dataframes = []

print(f"Scanning '{root_folder_path}' for data...")

# 1. TRAVERSE FOLDERS
# os.walk automatically looks inside 'Quebec ER Data' and every '2025-12-XX' folder inside it
for root, dirs, files in os.walk(root_folder_path):
    for filename in files:
        if filename.endswith(".csv"):
            file_path = os.path.join(root, filename)
            
            try:
                # 2. READ THE FILE
                # We use encoding='latin1' or 'utf-8' depending on the file. 
                # If you get "UnicodeDecodeError", change this to encoding='latin1'
                current_df = pd.read_csv(file_path, encoding='utf-8')
                
                # 3. STANDARDIZE TIMESTAMP
                # We rely on 'Mise_a_jour' because it contains the full date and time (ISO format)
                # The file name also has info, but the column is safer.
                if 'Mise_a_jour' in current_df.columns:
                    current_df['Timestamp'] = pd.to_datetime(current_df['Mise_a_jour'])
                else:
                    # Fallback: Create a timestamp from the folder name + file name if needed
                    # But your data likely has the column based on the sample you sent.
                    print(f"Warning: 'Mise_a_jour' missing in {filename}")
                
                # Add a column for the Source File just in case you need to debug later
                current_df['Source_File'] = filename
                
                all_dataframes.append(current_df)
                
            except Exception as e:
                print(f"Skipped {filename} due to error: {e}")

# 4. MERGE EVERYTHING
if all_dataframes:
    print(f"Merging {len(all_dataframes)} files...")
    master_df = pd.concat(all_dataframes, ignore_index=True)
    
    # 5. CLEANUP
    # Sort by hospital and then by time to make the data linear
    master_df.sort_values(by=['Nom_installation', 'Timestamp'], inplace=True)
    
    # Save to one big CSV
    output_filename = 'Quebec_ER_Master_Dataset.csv'
    master_df.to_csv(output_filename, index=False)
    
    print("------------------------------------------------")
    print(f"SUCCESS! Data saved to '{output_filename}'")
    print(f"Total rows collected: {len(master_df)}")
    print(f"Time range: {master_df['Timestamp'].min()} to {master_df['Timestamp'].max()}")
    print("------------------------------------------------")

else:
    print("No CSV files were found. Check your folder path.")