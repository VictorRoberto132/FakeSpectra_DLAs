import numpy as np
import pandas as pd
import os
import sys
import multiprocessing
from functools import partial
from tqdm import tqdm

# --- IMPORT CONFIGURATION ---
try:
    import utils
    import radii_skewers
except ImportError:
    # Fallback in case the script is called from outside the tools folder
    sys.path.append(os.path.abspath("."))
    import utils
    import radii_skewers

# --- 1. GENERAL CONFIGURATION ---
SNAP_NUM = "025"

# ==========================================
# MASTER KEY: Change only this to run other simulations
SIM_NAME = "TNG50-4"  # E.g.: "fiducial", "LH_0", "LH_1", etc.
# ==========================================

# --- PATH CONFIGURATION ---
# Gets the folder where this .py file is saved (the 'tools' folder)
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
# Defines the project root (one level above tools)
PROJ_ROOT = os.path.dirname(TOOLS_DIR)

# Adds the root to the Python Path
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

XMATCH_DIR = os.path.join(PROJ_ROOT, "xmatchs")

# Local directory setup (saving inside the project folder)
LIBRARY_DIR = os.path.join(PROJ_ROOT, f"notebooks/mini_sims_library_{SIM_NAME}")

if not os.path.exists(LIBRARY_DIR):
    print(f"--> [CRITICAL ERROR] Simulation folder '{SIM_NAME}' not found at:")
    print(f"    {LIBRARY_DIR}")
    sys.exit(1)

print(f"--> Processing Simulation: {SIM_NAME}")
print(f"--> Xmatch Directory     : {XMATCH_DIR}")
print(f"--> Library Directory    : {LIBRARY_DIR}")

# Total Halo Range (0 to 451)
ALL_MATCH_INDICES = list(range(0, 452))

# Radii Configuration
NUM_PARAM_IMPACTS = 2
RANGE_PARAM_IMPACTS = [0.1, 3.0]
R_MULTIPLIERS = np.logspace(np.log10(RANGE_PARAM_IMPACTS[0]), np.log10(RANGE_PARAM_IMPACTS[1]), NUM_PARAM_IMPACTS)

# Number of CPUs
NUM_CORES = 28

# --- 2. WORKER FUNCTION ---
def process_single_halo(match_index):
    """
    Processes an entire halo.
    """
    results_buffer = []
    
    try:
        # A. Initial Setup (SIM_NAME ensures it gets the correct xmatch file)
        matcher = utils.HaloMatcher(XMATCH_DIR)
        real_halo_id = matcher.get_halo_id(SIM_NAME, match_index)
        
        path_to_halo = os.path.join(LIBRARY_DIR, f"halo_{real_halo_id}")
        
        # Safety check: if the specific halo folder is not found, return empty
        if not os.path.exists(path_to_halo):
            return [] 

        # B. Load Context and Geometry
        ctx = utils.SimulationContext(path_to_halo, SNAP_NUM)
        halo_finder = utils.HaloFinder(ctx)
        
        # In the mini-simulation, the local halo is always 0
        halo_info = halo_finder.get_halo_properties(0)
        halo_info['id'] = real_halo_id 
        
        geom = radii_skewers.SkewerGeometry(ctx, halo_info)
        
        # C. Radii Loop
        for r_mult in R_MULTIPLIERS:
            # 1. Generate Skewers
            skewers_list = geom.generate_12_skewers(r_multiplier=r_mult)
            
            # 2. Run Fake Spectra 
            batch_results = geom.run_analysis(skewers_list, center_mode='gas')
            
            # 3. Extract Data
            for res in batch_results:
                summary_dict = {
                    # --- Identification ---
                    'match_index': match_index,
                    'halo_id': real_halo_id,
                    'halo_mass': res['halo_mass'],   # 10^10 Msun/h
                    'halo_r200_mpc': res['halo_r200'],
                    'redshift': res['redshift'],
                    
                    # --- Geometry ---
                    'skewer_id': res['id'],
                    'axis': res['axis'],             
                    'r_multiplier': r_mult,
                    'impact_kpc': res['impact_kpc'],
                    
                    # --- Basic Physics ---
                    'log_n_hi': res['log_n_hi'],
                    'n_hi_col': res['n_hi_col'],
                    
                    # --- Line Moments ---
                    'eq_width': res.get('eq_width', np.nan),
                    'std_dev': res.get('std_dev', np.nan),
                    'skewness': res.get('skewness', np.nan),
                    'kurtosis': res.get('kurtosis', np.nan),
                    
                    # --- Velocities ---
                    'mean_wave': res.get('mean_wave', np.nan),
                    'mean_vel': res.get('mean_vel', np.nan),
                    'vel_coherent': res.get('vel_coherent', np.nan),
                    
                    # --- FULL ARRAYS ---
                    'wave': res['wave'].astype(np.float32), 
                    'flux': res['flux'].astype(np.float32)
                }
                results_buffer.append(summary_dict)
                
    except Exception as e:
        # Print error
        print(f"[ERROR] Halo Index {match_index}: {e}")
        return []

    return results_buffer

# --- 3. MAIN BLOCK ---
if __name__ == "__main__":
    print(f"--> Starting parallel processing on {NUM_CORES} cores.")
    print(f"--> Total Halos: {len(ALL_MATCH_INDICES)}")
    
    all_data = []
    
    # Start the Pool
    with multiprocessing.Pool(processes=NUM_CORES) as pool:
        iterator = pool.imap_unordered(process_single_halo, ALL_MATCH_INDICES)
        
        for result_list in tqdm(iterator, total=len(ALL_MATCH_INDICES), unit="halo"):
            if result_list:
                all_data.extend(result_list)
                
    print("\n--> Processing complete. Generating DataFrame...")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    if not df.empty:
        df = df.sort_values(by=['match_index', 'r_multiplier', 'skewer_id'])
        
        # Dynamic output filename!
        output_filename = f"{SIM_NAME}_452Halos_FullSpectra.pkl"
        
        print(f"--> Saving {output_filename} (this might take a minute)...")
        df.to_pickle(output_filename)
        
        print(f"--> SUCCESS! File saved: {output_filename}")
        print(f"--> Memory footprint: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        print(f"--> Total rows: {len(df)}")
    else:
        print("--> CRITICAL ERROR: No data was processed.")