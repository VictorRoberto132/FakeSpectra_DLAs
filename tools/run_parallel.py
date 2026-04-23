import numpy as np
import pandas as pd
import os
import sys
import time
import logging
import multiprocessing
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

# --- TERMINAL COLORS ---
class TerminalUI:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# --- LOGGING CONFIGURATION ---
# Sends errors to a file so the terminal progress bar doesn't break
LOG_FILE = "extraction_errors.log"
logging.basicConfig(filename=LOG_FILE, filemode='w', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. GENERAL CONFIGURATION ---
SNAP_NUM = "094"  # Snapshot number to process
SIM_NAME = "LH_1" # Matches the updated xmatch and folder names
BASE_PATH = "/home/STORAGE/MN5_resims/LH_1/hydro_output"
SIM_TYPE = "MN5" # Set to "MN5" or "TNG" to handle kernel assignment

# Set the number of Halos to process. Use an integer (e.g., 1) or "all"
N_HALOS_TO_PROCESS = "all"

# --- PATH CONFIGURATION ---
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(TOOLS_DIR)

if PROJ_ROOT not in sys.path: 
    sys.path.insert(0, PROJ_ROOT)

XMATCH_DIR = os.path.join(PROJ_ROOT, "xmatchs")

# Total Halo Range (Adjust according to your cross-match catalog size)
ALL_MATCH_INDICES = list(range(0, 452))

# Apply the user-defined limit
if str(N_HALOS_TO_PROCESS).lower() != "all":
    ALL_MATCH_INDICES = ALL_MATCH_INDICES[:int(N_HALOS_TO_PROCESS)]

# Radii Configuration
NUM_PARAM_IMPACTS = 15
RANGE_PARAM_IMPACTS = [0.01, 3.0]

# Generate log-spaced array and insert 0.0 at the beginning
log_steps = np.logspace(np.log10(RANGE_PARAM_IMPACTS[0]), np.log10(RANGE_PARAM_IMPACTS[1]), NUM_PARAM_IMPACTS)
R_MULTIPLIERS = np.insert(log_steps, 0, 0.0)

NUM_CORES = 21  # User specified parallel capacity

def estimate_time(n_halos, n_cores):
    """Provides a rough physical time prescription for the full map pipeline."""
    # 3 skewers for r=0, and 12 skewers for the remaining 5 radii = 63 skewers per halo
    n_skewers = 3 + (12 * NUM_PARAM_IMPACTS)
    # Assuming the full map KD-Tree + SPH mapping takes ~1.5 seconds per skewer 
    # due to the full BoxSize search tree logic.
    time_per_halo_sec = n_skewers * 90  
    total_time_sec = (n_halos * time_per_halo_sec) / n_cores
    return total_time_sec / 60.0

def align_to_absorption_center(vel_array, flux_array):
    """
    Encontra o centro real da absorção. Protegido contra múltiplos sistemas saturados
    na mesma linha de visada, escolhendo a bacia mais próxima do centro cinemático.
    """
    min_flux = np.min(flux_array)
    threshold = min_flux + 0.02 
    
    # Encontra todos os índices no fundo do poço
    bottom_indices = np.where(flux_array <= threshold)[0]
    
    if len(bottom_indices) == 0:
        return vel_array, 0.0
        
    # Identifica se existem quebras/buracos entre os índices (múltiplas absorções separadas)
    jumps = np.where(np.diff(bottom_indices) > 1)[0]
    
    if len(jumps) == 0:
        # Cenário seguro: Apenas uma bacia contínua encontrada
        v_center = np.mean(vel_array[bottom_indices])
    else:
        # Cenário complexo: Duas ou mais bacias saturadas empatadas!
        # Separa os índices em blocos (starts e ends)
        starts = np.insert(bottom_indices[jumps + 1], 0, bottom_indices[0])
        ends = np.append(bottom_indices[jumps], bottom_indices[-1])
        
        best_center = 0.0
        min_dist_to_kinematic_zero = np.inf
        
        # Calcula o centro de cada bacia isoladamente
        for s, e in zip(starts, ends):
            island_indices = np.arange(s, e + 1)
            island_v_center = np.mean(vel_array[island_indices])
            
            # Elege como "v_center" a bacia que estiver fisicamente mais associada
            # ao halo central original (mais próxima do v=0 original)
            if np.abs(island_v_center) < min_dist_to_kinematic_zero:
                min_dist_to_kinematic_zero = np.abs(island_v_center)
                best_center = island_v_center
                
        v_center = best_center
        
    vel_aligned = vel_array - v_center
    return vel_aligned, v_center

# --- 2. WORKER FUNCTION ---
def process_single_halo(match_index):
    """
    Processes an entire halo direct from the full map.
    Returns: (match_index, real_halo_id, results_buffer, status)
    """
    results_buffer = []
    real_halo_id = "UNKNOWN"
    
    try:
        # Resolve real ID from cross-match
        matcher = utils.HaloMatcher(XMATCH_DIR)
        real_halo_id = matcher.get_halo_id(SIM_NAME, match_index)
        
        # Load Context (pointing directly to the full simulation)
        ctx = utils.SimulationContext(BASE_PATH, SNAP_NUM, sim_type=SIM_TYPE)
        halo_finder = utils.HaloFinder(ctx)
        
        # Grabs properties from the master group catalogs
        halo_info = halo_finder.get_halo_properties(real_halo_id)
        
        geom = radii_skewers.SkewerGeometry(ctx, halo_info)
        
        all_skewers = []
        for r_mult in R_MULTIPLIERS:
            all_skewers.extend(geom.generate_skewers(r_multiplier=r_mult))

        batch_results = geom.run_analysis_batched(all_skewers)

        for res in batch_results:
            # O ALINHAMENTO DEVE ACONTECER AQUI DENTRO DO LOOP
            # E precisamos receber as duas variáveis de volta
            vel_aligned_arr, v_center_val = align_to_absorption_center(res['vel_axis'], res['flux'])

            summary_dict = {
                'match_index': match_index,
                'halo_id': real_halo_id,
                'halo_mass': res['halo_mass'],
                'halo_r200_mpc': res['halo_r200'],
                'redshift': res['redshift'],
                
                'skewer_id': res['id'],
                'axis': res['axis'],             
                'r_multiplier': res['r_mult'],
                'impact_kpc': res['impact_kpc'],
                
                'log_n_hi': res['log_n_hi'],
                'total_col_den': res['total_col_den'],
                
                'eq_width': res.get('eq_width', np.nan),
                'std_dev': res.get('std_dev', np.nan),
                'skewness': res.get('skewness', np.nan),
                'kurtosis': res.get('kurtosis', np.nan),
                
                'mean_wave': res.get('mean_wave', np.nan),
                'mean_vel': res.get('mean_vel', np.nan),
                'vel_coherent': res.get('vel_coherent', np.nan),
                
                # Valor numérico do quanto o espectro foi movido
                'vel_peak_shift': v_center_val, 
                
                # Store fully centered native grids
                'wave': res['wave'].astype(np.float32), 
                'flux': res['flux'].astype(np.float32),
                'vel_axis': res['vel_axis'].astype(np.float32),
                'vel_aligned': vel_aligned_arr.astype(np.float32),
                'chi_axis': res['chi_axis'].astype(np.float32),
                'n_hi_col': res['n_hi_col'].astype(np.float32)
            }
            results_buffer.append(summary_dict)
                
    except Exception as e:
        # Fails safely and logs the traceback
        logging.error(f"Halo Index {match_index} (ID {real_halo_id}): {str(e)}")
        return (match_index, real_halo_id, [], "ERROR")

    return (match_index, real_halo_id, results_buffer, "SUCCESS")


# --- 3. MAIN BLOCK ---
if __name__ == "__main__":
    if os.path.exists(LOG_FILE): 
        os.remove(LOG_FILE)

    n_halos = len(ALL_MATCH_INDICES)
    est_mins = estimate_time(n_halos, NUM_CORES)

    print(f"\n{TerminalUI.CYAN}{TerminalUI.BOLD}======================================================================{TerminalUI.RESET}")
    print(f"{TerminalUI.CYAN}{TerminalUI.BOLD}   FULL-MAP SKEWER EXTRACTION (DLA MODE - STRUCTURAL KD-TREE){TerminalUI.RESET}")
    print(f"{TerminalUI.CYAN}{TerminalUI.BOLD}======================================================================{TerminalUI.RESET}")
    print(f"{TerminalUI.BOLD}--> Target Map        :{TerminalUI.RESET} {TerminalUI.GREEN}{SIM_NAME} (Snap {SNAP_NUM}){TerminalUI.RESET}")
    print(f"{TerminalUI.BOLD}--> Base Path         :{TerminalUI.RESET} {BASE_PATH}")
    print(f"{TerminalUI.BOLD}--> Simulation Type   :{TerminalUI.RESET} {SIM_TYPE}")
    print(f"{TerminalUI.BOLD}--> Physical Cores    :{TerminalUI.RESET} {NUM_CORES}")
    print(f"{TerminalUI.BOLD}--> Total Halos       :{TerminalUI.RESET} {n_halos}")
    print(f"{TerminalUI.WARNING}{TerminalUI.BOLD}--> Estimated Time    : ~{est_mins:.1f} minutes{TerminalUI.RESET}")
    print(f"{TerminalUI.CYAN}----------------------------------------------------------------------{TerminalUI.RESET}\n")
    
    all_data = []
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    bar_fmt = f"{TerminalUI.GREEN}{{l_bar}}{{bar:40}}{TerminalUI.RESET}| {{n_fmt}}/{{total_fmt}} halos [{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"

    with multiprocessing.Pool(processes=NUM_CORES) as pool:
        iterator = pool.imap_unordered(process_single_halo, ALL_MATCH_INDICES)
        
        with tqdm(total=n_halos, bar_format=bar_fmt) as pbar:
            for match_idx, real_id, result_list, status in iterator:
                
                if status == "SUCCESS" and result_list:
                    all_data.extend(result_list)
                    success_count += 1
                else:
                    fail_count += 1
                
                pbar.set_postfix({
                    "Last ID": f"{real_id:<6}", 
                    "Success": success_count, 
                    "Skips/Errs": fail_count
                })
                pbar.update(1)
                
    print(f"\n\n{TerminalUI.CYAN}----------------------------------------------------------------------{TerminalUI.RESET}")
    print(f"{TerminalUI.BOLD}--> Processing complete. Generating DataFrame...{TerminalUI.RESET}")
    
    df = pd.DataFrame(all_data)
    
    if not df.empty:
        df = df.sort_values(by=['match_index', 'r_multiplier', 'skewer_id'])
        
        # Name adjusts based on how many halos we processed
        tag = f"{n_halos}Halos" if str(N_HALOS_TO_PROCESS).lower() != "all" else "AllHalos"
        output_filename = f"{SIM_NAME}_{tag}_DLAMode_FullMap.pkl"
        output_path = os.path.join(PROJ_ROOT, "notebooks", output_filename)
        
        print(f"{TerminalUI.BOLD}--> Saving DataFrame...{TerminalUI.RESET}")
        df.to_pickle(output_path)
        
        elapsed = (time.time() - start_time) / 60
        
        print(f"\n{TerminalUI.GREEN}{TerminalUI.BOLD}[✔] SUCCESS! Pipeline Finished in {elapsed:.1f} minutes.{TerminalUI.RESET}")
        print(f"{TerminalUI.BOLD}    Saved To    :{TerminalUI.RESET} {output_path}")
        print(f"{TerminalUI.BOLD}    Total Rows  :{TerminalUI.RESET} {len(df):,}")
        print(f"{TerminalUI.BOLD}    Memory Size :{TerminalUI.RESET} {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        
        if fail_count > 0:
            print(f"{TerminalUI.WARNING}{TerminalUI.BOLD}[!] NOTE: {fail_count} halos had errors/skips. Check '{LOG_FILE}'.{TerminalUI.RESET}")
    else:
        print(f"\n{TerminalUI.FAIL}{TerminalUI.BOLD}[✘] CRITICAL ERROR: No data was processed. Check '{LOG_FILE}'.{TerminalUI.RESET}")