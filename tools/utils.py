import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import sys
import logging
from matplotlib.colors import LogNorm 
import fake_spectra.spectra as spectra
import fake_spectra.abstractsnapshot as absn

# --- LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# CLASS 0: HALO MATCHER
# ==============================================================================
class HaloMatcher:
    def __init__(self, xmatch_dir):
        self.xmatch_dir = xmatch_dir
        self.match_catalogs = {}

    def get_halo_id(self, sim_name, match_index):
        if sim_name not in self.match_catalogs:
            filename = f"xmatch_{sim_name}.npy"
            path = os.path.join(self.xmatch_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Match file not found at: {path}")
            
            logging.info(f"Loading match catalog for {sim_name}...")
            data = np.load(path, allow_pickle=True)
            if data.ndim == 0:
                self.match_catalogs[sim_name] = data.item()['ind']
            else:
                self.match_catalogs[sim_name] = data[0]['ind']

        catalog = self.match_catalogs[sim_name]
        if match_index < 0 or match_index >= len(catalog):
            raise ValueError(f"Match Index {match_index} is out of bounds.")
            
        real_id = catalog[match_index]
        return int(real_id)

# ==============================================================================
# CLASS 1: SIMULATION CONTEXT
# ==============================================================================
class SimulationContext:
    def __init__(self, base_path, snap_num):
        self.base_path = base_path
        self.snap_num = snap_num
        self.group_dir = f"groups_{snap_num}"
        self.snap_dir = f"snapdir_{snap_num}"
        self._apply_mn5_patch()
        self.baryonic_params = self._load_baryonic_params()

    def _apply_mn5_patch(self):
        """Patches fake_spectra to handle MN5 specific HDF5 attributes."""
        TargetClass = absn.AbstractSnapshot
        if not hasattr(TargetClass, '_original_get_header_attr'):
            TargetClass._original_get_header_attr = TargetClass.get_header_attr

        def patched_get_header_attr(instance, attr):
            try:
                return instance._original_get_header_attr(attr)
            except KeyError:
                pass 
            if hasattr(instance, '_f_handle') and "Parameters" in instance._f_handle:
                try:
                    return instance._f_handle["Parameters"].attrs[attr]
                except KeyError:
                    pass
            if attr == "NumPart_Total_HighWord": return 0
            raise KeyError(f"Could not find attribute '{attr}'")

        TargetClass.get_header_attr = patched_get_header_attr

    def _load_baryonic_params(self):
        g_file = os.path.join(self.base_path, self.group_dir, f"fof_subhalo_tab_{self.snap_num}.0.hdf5")
        keys = ["WindEnergyIn1e51erg", "VariableWindVelFactor", "WindFreeTravelDensFac",
                "MaxSfrTimescale", "RadioFeedbackFactor", "BlackHoleFeedbackFactor", 
                "RadioFeedbackReiorientationFactor"]
        params = {}
        try:
            with h5py.File(g_file, 'r') as f:
                if 'Parameters' in f:
                    attrs = f['Parameters'].attrs
                    for k in keys: params[k] = attrs.get(k, np.nan)
        except Exception: pass
        return params

    def get_group_file(self, index):
        return os.path.join(self.base_path, self.group_dir, f"fof_subhalo_tab_{self.snap_num}.{index}.hdf5")

    def get_snap_file(self, index):
        return os.path.join(self.base_path, self.snap_dir, f"snapshot_{self.snap_num}.{index}.hdf5")

# ==============================================================================
# CLASS 2: HALO FINDER (UPDATED FOR GROUPS/FOF)
# ==============================================================================
class HaloFinder:
    def __init__(self, sim_context):
        self.sim = sim_context

    def get_halo_properties(self, global_halo_id):
        """
        Retrieves properties for a FOF GROUP (Halo), not a Subhalo.
        """
        # Read header from the first file to get totals
        with h5py.File(self.sim.get_group_file(0), 'r') as f:
            n_group_files = int(f['Header'].attrs['NumFiles'])
            box_size = f['Header'].attrs['BoxSize']
            n_total = int(f['Header'].attrs['Ngroups_Total']) 
            scale_factor = f['Header'].attrs['Time'] 
            redshift = f['Header'].attrs['Redshift']

        if global_halo_id >= n_total: 
            raise ValueError(f"Halo ID {global_halo_id} out of bounds (Total Groups: {n_total}).")

        halo_count = 0
        for i in range(n_group_files):
            g_file = self.sim.get_group_file(i)
            with h5py.File(g_file, 'r') as f:
                n_here = int(f['Header'].attrs['Ngroups_ThisFile'])
                
                if halo_count + n_here > global_halo_id:
                    local_index = int(global_halo_id - halo_count)
                    
                    # Assume standard TNG units for velocity: km/s/a.
                    raw_vel = f['Group']['GroupVel'][local_index]
                    phys_vel = raw_vel * np.sqrt(scale_factor) 
                    
                    r200 = 0.0
                    if 'Group_R_Crit200' in f['Group']:
                        r200 = f['Group']['Group_R_Crit200'][local_index]
                        
                    return {
                        'id': global_halo_id,
                        'cm': f['Group']['GroupPos'][local_index],
                        'mass': f['Group']['Group_M_Crit200'][local_index],
                        'r200': r200, 
                        'vel_pec': phys_vel,
                        'box_size': box_size,
                        'redshift': redshift,
                        'scale_factor': scale_factor
                    }
                halo_count += n_here
        raise ValueError(f"Halo {global_halo_id} not found in any group file.")

# ==============================================================================
# CLASS 3: GAS LOADER
# ==============================================================================
class GasLoader:
    def __init__(self, sim_context):
        self.sim = sim_context

    @staticmethod
    def _apply_pbc_shift(coords, center, box):
        delta = coords - center
        delta[delta > box/2] -= box
        delta[delta < -box/2] += box
        return delta

    def load_gas_slice(self, halo_cm, box_size, slice_thickness=10.0):
        gas_coords = []
        with h5py.File(self.sim.get_snap_file(0), 'r') as f:
            n_files = int(f['Header'].attrs['NumFilesPerSnapshot'])

        for i in range(n_files):
            try:
                with h5py.File(self.sim.get_snap_file(i), 'r') as f:
                    if 'PartType0' in f:
                        pos = f['PartType0']['Coordinates'][:]
                        dz = pos[:, 2] - halo_cm[2]
                        dz[dz > box_size/2] -= box_size
                        dz[dz < -box_size/2] += box_size
                        mask = (np.abs(dz) < slice_thickness/2)
                        if np.sum(mask) > 0:
                            centered = self._apply_pbc_shift(pos[mask], halo_cm, box_size)
                            gas_coords.append(centered)
            except: continue
        return np.concatenate(gas_coords) if gas_coords else np.zeros((1,3))

# ==============================================================================
# CLASS 4: SPECTRUM & MULTI-SKEWER ENGINES
# ==============================================================================
# ... (imports permanecem iguais)

class SpectrumEngine:
    def __init__(self, sim_context, halo_data):
        self.sim = sim_context
        self.halo = halo_data
        self.ss = None
        self.flux = None
        self.axis = 1
        self.tau = None
        # Stores the scalar Column Density (summed inside isolation radius)
        self.n_hi_masked = 0.0 
        self.tau_threshold = 0.1 
        # [NOVO] Armazena o array de densidade para cálculo de velocidade coerente
        self.col_den_centered = None 
        
    def generate_skewer(self, axis=1):
        """Generates the full box skewer using fake_spectra."""
        self.axis = axis 
        cofm = np.array([self.halo['cm']]); axis_arr = np.array([axis], dtype=np.int32)
        save_file = f"temp_skewer_{self.halo['id']}_{np.random.randint(100000)}.hdf5"
        
        # Reload_file=True forces recalculation (no caching issues)
        self.ss = spectra.Spectra(self.sim.snap_num, self.sim.base_path, cofm=cofm, axis=axis_arr, 
                                  res=10, savefile=save_file, reload_file=True)
        
        # Get Optical Depth (Tau)
        self.tau = self.ss.get_tau("H", 1, 1215)[0, :]
        
        self.flux = None 
        if os.path.exists(save_file): os.remove(save_file)
        return self.tau

    def get_centered_spectrum(self, mode='halo', window_radius_kms=1500.0, isolation_radius_mpc=None):
        """
        Centers the spectrum and optionally masks absorption outside a given physical radius.
        Also calculates the N_HI (Column Density) within that mask.
        """
        if self.tau is None: raise RuntimeError("Run generate_skewer() first.")

        # --- Get Column Density from fake_spectra (Shape: [1, nbins]) ---
        # Units: ions / cm^2 per pixel
        col_den_raw = self.ss.get_col_density("H", 1)[0, :]

        nbins = self.ss.nbins
        target_center = nbins // 2
        
        # --- 1. Determine Shift needed to center the Halo ---
        axis_idx = self.axis - 1 
        v_pec = self.halo['vel_pec'][axis_idx]
        v_hub = (self.halo['cm'][axis_idx] / self.halo['box_size']) * self.ss.vmax
        halo_px = int(np.round(((v_hub + v_pec) % self.ss.vmax) / self.ss.vmax * nbins))
        
        shift_bins = 0

        if mode == 'halo':
            shift_bins = target_center - halo_px
            
        elif mode == 'gas':
            # Logic to center on the deepest absorption peak
            temp_shift = target_center - halo_px
            tau_temp = np.roll(self.tau, temp_shift)
            px_kms = self.ss.vmax / nbins
            win_px = int(window_radius_kms / px_kms)
            start_win = max(0, target_center - win_px)
            end_win = min(nbins, target_center + win_px)
            window_slice = tau_temp[start_win:end_win]
            
            if np.max(window_slice) > self.tau_threshold:
                local_peak_idx = np.argmax(window_slice)
                global_peak_idx = start_win + local_peak_idx
                left_edge = global_peak_idx
                while left_edge > 0 and tau_temp[left_edge] > self.tau_threshold:
                    left_edge -= 1
                right_edge = global_peak_idx
                while right_edge < nbins - 1 and tau_temp[right_edge] > self.tau_threshold:
                    right_edge += 1
                center_mass_px = (left_edge + right_edge) / 2.0
                fine_adjustment = target_center - center_mass_px
                shift_bins = int(np.round(temp_shift + fine_adjustment))
            else:
                shift_bins = temp_shift

        # Apply the roll to center the structure (Tau and Column Density)
        tau_centered = np.roll(self.tau, shift_bins)
        
        # [NOVO] Guardamos no self para usar no radii_skewers
        self.col_den_centered = np.roll(col_den_raw, shift_bins) 

        # --- 2. Apply Isolation Mask ---
        if isolation_radius_mpc is not None:
            box_mpc = self.halo['box_size']
            vel_radius_kms = (isolation_radius_mpc / box_mpc) * self.ss.vmax
            px_kms = self.ss.vmax / nbins
            radius_px = int(vel_radius_kms / px_kms)
            
            indices = np.arange(nbins)
            dist_from_center = np.abs(indices - target_center)
            
            # Mask: True inside the region, False outside
            mask = dist_from_center <= radius_px
            
            # Apply mask to Tau (set to 0 => Flux=1)
            tau_centered[~mask] = 0.0
            
            # Apply mask to Column Density (set to 0)
            self.col_den_centered[~mask] = 0.0
            
            # Calculate Total N_HI within the isolation radius
            self.n_hi_masked = np.sum(self.col_den_centered)
            
            # logging.info(f"Isolation applied: R={isolation_radius_mpc:.2f} Mpc/h.")
        else:
            # If no mask, sum the whole box
            self.n_hi_masked = np.sum(self.col_den_centered)

        # --- 3. Compute Flux and Axis ---
        self.flux = np.exp(-tau_centered)
        
        c_light = 299792.458; lam_rest = 1215.67
        v_axis = (np.arange(nbins) - target_center) * (self.ss.vmax / nbins)
        wave_axis = lam_rest * (1 + v_axis / c_light)
        
        return wave_axis, self.flux, lam_rest