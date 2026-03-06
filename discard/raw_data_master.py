import pandas as pd
import glob
import os

def compile_er_data(source_folder='raw_data', output_file='master_dataset.csv'):
    # 1. Get a list of all CSV files in the folder
    files = glob.glob(os.path.join(source_folder, "*.csv"))
    print(f"Found {len(files)} files in {source_folder}. Starting process...")

    all_data = []
    files_processed = 0
    files_skipped = 0

    # 2. Loop through each file
    for filename in files:
        try:
            # Read the CSV file
            # on_bad_lines='skip' helps skip broken lines in corrupted files
            df = pd.read_csv(filename, on_bad_lines='skip')

            # Check if file is empty or missing critical columns
            if df.empty:
                files_skipped += 1
                continue
            
            # Optional: Check for a critical column to ensure it's a valid ER file
            # Adjust 'Mise_a_jour' if your files have different headers
            if 'Mise_a_jour' not in df.columns:
                files_skipped += 1
                continue

            # Append to our list
            all_data.append(df)
            files_processed += 1

        except Exception as e:
            # Handle completely unusable/corrupt files
            print(f"Error reading {filename}: {e}")
            files_skipped += 1

    if not all_data:
        print("No valid data found. Check your folder path.")
        return

    print(f"Concatenating {files_processed} valid files...")
    
    # 3. Combine all dataframes into one
    master_df = pd.concat(all_data, ignore_index=True)

    # 4. Remove Duplicates
    # Since the scraper runs every 10 mins but data updates hourly, 
    # many rows will be identical. dropping duplicates removes these redundant snapshots.
    initial_rows = len(master_df)
    master_df.drop_duplicates(inplace=True)
    final_rows = len(master_df)

    # 5. Sort by time
    # Convert 'Mise_a_jour' to datetime for sorting (and for your analysis)
    if 'Mise_a_jour' in master_df.columns:
        master_df['Mise_a_jour'] = pd.to_datetime(master_df['Mise_a_jour'], errors='coerce')
        master_df.sort_values(by=['Mise_a_jour', 'Nom_etablissement'], inplace=True)

    # 6. Save the result
    master_df.to_csv(output_file, index=False)
    
    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Files processed: {files_processed}")
    print(f"Files skipped (empty/error): {files_skipped}")
    print(f"Total rows read: {initial_rows}")
    print(f"Unique rows after deduplication: {final_rows}")
    print(f"Duplicates removed: {initial_rows - final_rows}")
    print(f"Saved master dataset to: {output_file}")

# Run the function
if __name__ == "__main__":
    compile_er_data()