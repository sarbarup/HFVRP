"""
Runner script to execute the time-classes HFVRP model and print/save the resulting tables.

Usage:
1. Make sure `Final_time_classes.py` (the module implementing the model) is saved in the same directory.
   If you don't have it, save the file I previously provided as `Final_time_classes.py`.

2. Install dependencies (example for a local environment or Colab):
   pip install ortools pandas numpy

3. Run this script:
   python run_test_time_classes.py

What it does:
- Imports solve_with_time_classes and reporting helpers from Final_time_classes.py
- Runs the solver with a configurable time limit (default 10 seconds)
- Builds routes_df and visits_df (Pandas DataFrames)
- Prints a concise summary and the full DataFrames (or the head if very large)
- Saves routes_summary.csv and visits_detail.csv in the current directory
- Catches and reports solver failures or import errors with actionable messages
"""
import sys
import traceback
import os

# Configurable parameters
TIME_LIMIT_SECONDS = 10   # increase to give the solver more time (e.g., 30, 60)
SAVE_CSV = True           # set to False to avoid writing CSV files
MAX_ROWS_PRINT = 200      # avoid extremely long terminal prints; adjust as needed

def main():
    try:
        # Try to import the model module that implements the time-class approach
        import Final_time_classes as model
    except Exception as e:
        print("ERROR: Could not import Final_time_classes module.")
        print("Make sure Final_time_classes.py is in the same directory as this runner.")
        print("If you don't have the file, save the 'Final_time_classes.py' provided earlier in this conversation.")
        print("Import error detail:")
        traceback.print_exc()
        sys.exit(1)

    try:
        print(f"Running solver (time limit = {TIME_LIMIT_SECONDS}s). This may take a few seconds...")
        data, manager, routing, solution, time_dims = model.solve_with_time_classes(time_limit_seconds=TIME_LIMIT_SECONDS)
        print("Solver completed successfully.\n")
    except Exception as e:
        print("Solver failed or raised an exception.")
        traceback.print_exc()
        sys.exit(1)

    try:
        routes_df, visits_df = model.create_solution_tables(data, manager, routing, solution, time_dims)
    except Exception as e:
        print("Failed to create solution tables from the solver output.")
        traceback.print_exc()
        sys.exit(1)

    # Print summary
    try:
        print("\n=== ROUTE SUMMARY DATAFRAME ===\n")
        # If not too large, print full DataFrame; otherwise print head()
        if len(routes_df) <= MAX_ROWS_PRINT:
            print(routes_df.to_string(index=False))
        else:
            print(routes_df.head(50).to_string(index=False))
            print(f"\n... (DataFrame has {len(routes_df)} rows; showing head only)")

        print("\n=== VISIT DETAILS DATAFRAME (first 200 rows) ===\n")
        if len(visits_df) <= MAX_ROWS_PRINT:
            print(visits_df.to_string(index=False))
        else:
            print(visits_df.head(200).to_string(index=False))
            print(f"\n... (DataFrame has {len(visits_df)} rows; showing head only)")

        # High-level numeric summary
        try:
            total_distance_km = routes_df['Total_Distance_km'].sum()
            total_time_hours = routes_df['Total_Time_hours'].sum()
            total_cost = routes_df['Total_Cost_$'].sum()
            total_load = routes_df['Total_Load_kg'].sum()
            vehicles_used = len(routes_df)
            total_stops = int(routes_df['Stops'].sum())
            print("\n=== AGGREGATE METRICS ===")
            print(f"Vehicles used: {vehicles_used} / {model.NUM_VEHICLES}")
            print(f"Total stops: {total_stops}")
            print(f"Total distance: {total_distance_km:,.1f} km")
            print(f"Total time: {total_time_hours:,.2f} hours")
            print(f"Total cost (fuel + labor): ${total_cost:,.2f}")
            print(f"Total load delivered: {total_load:,.0f} kg")
        except Exception:
            # if any column names differ or data missing
            print("Could not compute aggregate metrics (check DataFrame columns).")
    except Exception:
        print("Error printing DataFrames:")
        traceback.print_exc()

    # Save to CSV files
    if SAVE_CSV:
        try:
            routes_fname = "routes_summary_time_classes.csv"
            visits_fname = "visits_detail_time_classes.csv"
            routes_df.to_csv(routes_fname, index=False)
            visits_df.to_csv(visits_fname, index=False)
            print(f"\nSaved routes summary to: {os.path.abspath(routes_fname)}")
            print(f"Saved visits detail to: {os.path.abspath(visits_fname)}")
        except Exception:
            print("Failed to save CSVs:")
            traceback.print_exc()

if __name__ == "__main__":
    main()