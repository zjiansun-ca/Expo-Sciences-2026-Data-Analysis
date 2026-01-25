import pandas as pd

def check_master_dataset(file_path='master_dataset.csv'):
    print(f"Loading {file_path}...\n")
    
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Check if the critical date column exists
        if 'Mise_a_jour' not in df.columns:
            print("ERROR: Column 'Mise_a_jour' not found. Cannot proceed with time analysis.")
            return

        # Convert to datetime objects
        df['Timestamp'] = pd.to_datetime(df['Mise_a_jour'])
        
        # --- 1. DATE RANGE ---
        min_date = df['Timestamp'].min()
        max_date = df['Timestamp'].max()
        duration = max_date - min_date
        
        print("-" * 40)
        print(f"1. TIME RANGE")
        print("-" * 40)
        print(f"Start: {min_date}")
        print(f"End:   {max_date}")
        print(f"Total Duration: {duration}")
        print(f"Total Data Rows: {len(df):,}")
        print("\n")

        # --- 2. UNIQUE SNAPSHOTS ---
        # How many distinct moments in time do we have data for?
        unique_timestamps = df['Timestamp'].nunique()
        expected_hours = (duration.total_seconds() / 3600)
        
        print("-" * 40)
        print(f"2. COMPLETENESS")
        print("-" * 40)
        print(f"Total Unique Hourly Snapshots Found: {unique_timestamps}")
        # Note: This is an approximation. 
        print(f"Approximate Hours in Range: {int(expected_hours)}")
        print("\n")

        # --- 3. HOURLY DISTRIBUTION ---
        # This answers: "Do we have data for 3 AM as often as 3 PM?"
        df['Hour'] = df['Timestamp'].dt.hour
        df['Date_Str'] = df['Timestamp'].dt.date.astype(str)
        
        # We count how many unique DAYS appear for each HOUR.
        # e.g., If we have 10 days of data, we expect '10' for every hour.
        hourly_coverage = df.groupby('Hour')['Date_Str'].nunique()

        print("-" * 40)
        print(f"3. HOURLY COVERAGE (Count of Days with Data per Hour)")
        print("-" * 40)
        print("Hour | Days Covered | Status")
        print("-----|--------------|-------")
        
        max_days = hourly_coverage.max()
        
        for hour in range(24):
            count = hourly_coverage.get(hour, 0)
            # Visual indicator of missingness
            if count == 0:
                status = "MISSING ALL DATA" 
            elif count < max_days:
                status = f"MISSING {max_days - count} DAYS"
            else:
                status = "OK"
                
            print(f" {hour:02d}h | {count:12d} | {status}")
            
        # --- 4. PERTINENT EXTRAS ---
        print("\n")
        print("-" * 40)
        print(f"4. OTHER STATS")
        print("-" * 40)
        print(f"Unique Hospitals/Facilities: {df['Nom_etablissement'].nunique()}")
        print(f"Regions Covered: {df['Region'].nunique()}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{file_path}'. Make sure it is in the same folder.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    check_master_dataset()