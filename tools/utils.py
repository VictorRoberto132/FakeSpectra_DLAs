import h5py
import numpy as np
import os
import logging
import fake_spectra.spectra as spectra
import fake_spectra.abstractsnapshot as absn

# --- LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HaloMatcher:
    """Matches the catalog index to the global simulation Halo ID."""
    def __init__(self, xmatch_dir):
        self.xmatch_dir = xmatch_dir
        self.match_catalogs = {}

    def get_halo_id(self, sim_name, match_index):
        """Retrieves the real Halo ID given the cross-match index."""
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
            
        return int(catalog[match_index])


class SimulationContext:
    """Holds the global state and paths for the full simulation map."""
    def __init__(self, base_path, snap_num, sim_type="MN5"):
        self.base_path = base_path
        self.sim_type = sim_type.upper()
        
        self.snap_num = f"{snap_num:03d}" if isinstance(snap_num, int) else snap_num
        self.group_dir = f"groups_{self.snap_num}"
        self.snap_dir = f"snapdir_{self.snap_num}"
        
        # Apply MN5 specific patch to bypass missing HighWord attributes
        if self.sim_type == "MN5":
            self._apply_mn5_patch()

    def _apply_mn5_patch(self):
        """Monkey patches the absn get_header_attr to handle MN5 missing keywords."""
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
            if attr == "NumPart_Total_HighWord": 
                return np.zeros(6, dtype=np.int32)
            raise KeyError(f"Could not find attribute '{attr}'")

        TargetClass.get_header_attr = patched_get_header_attr

    def get_group_file(self, index):
        """Constructs the path for a specific FoF group file."""
        return os.path.join(self.base_path, self.group_dir, f"fof_subhalo_tab_{self.snap_num}.{index}.hdf5")


class HaloFinder:
    """Finds specific halos within the full simulation group catalog."""
    def __init__(self, sim_context):
        self.sim = sim_context

    def get_halo_properties(self, global_halo_id):
        """Scans the FoF files to find the center of mass, mass, and velocity of a target halo."""
        with h5py.File(self.sim.get_group_file(0), 'r') as f:
            n_group_files = int(f['Header'].attrs.get('NumFiles', f['Header'].attrs.get('NumFilesPerSnapshot', 1)))
            box_size = f['Header'].attrs['BoxSize']
            scale_factor = f['Header'].attrs['Time'] 
            redshift = f['Header'].attrs['Redshift']

        current_id_offset = 0
        for i in range(n_group_files):
            g_file = self.sim.get_group_file(i)
            if not os.path.exists(g_file): continue
                
            with h5py.File(g_file, 'r') as f:
                if 'Group' not in f or 'Group_M_Crit200' not in f['Group']: continue
                    
                n_here = len(f['Group']['Group_M_Crit200'])
                if global_halo_id < current_id_offset + n_here:
                    local_index = int(global_halo_id - current_id_offset)
                    return {
                        'id': global_halo_id,
                        'cm': f['Group']['GroupPos'][local_index],
                        'mass': f['Group']['Group_M_Crit200'][local_index],
                        'r200': f['Group']['Group_R_Crit200'][local_index], 
                        'vel_raw': f['Group']['GroupVel'][local_index],
                        'box_size': box_size,
                        'redshift': redshift,
                        'scale_factor': scale_factor
                    }
                current_id_offset += n_here
        raise ValueError(f"Halo {global_halo_id} not found in any group file.")


class SpectrumEngine:
    """Handles fake_spectra extraction natively in batch mode using RAM optimization."""
    def __init__(self, sim_context, halo_data):
        self.sim = sim_context
        self.halo = halo_data
        self.ss = None
        self.tau_batch = None
        self.col_den_batch = None
        
        # Cosmology Constants
        self.c_light = 299792.458 # km/s
        self.lam_rest = 1215.67   # Angstroms

    def generate_skewer_batch(self, axis_array, cofm_array):
        """
        Generates multiple skewers simultaneously applying native 1D Region Growing topological masking.
        This forces fake_spectra to read the full HDF5 simulation into RAM only once per halo.
        """
        # Generate a unique temp file to avoid collisions in parallel execution
        save_file = f"temp_batch_{self.halo['id']}_{np.random.randint(100000)}.hdf5"
        
        # Both MN5 and TNG are Arepo based. Kernel auto-detect (None) is optimal.
        self.ss = spectra.Spectra(
            int(self.sim.snap_num), 
            self.sim.base_path, 
            cofm=cofm_array, 
            axis=axis_array, 
            res=10.0,                      
            savefile=save_file, 
            reload_file=True,
            kernel=None,                  # AUTO-DETECT (Arepo Voronoi/Tophat)
            MPI=None,                     
            quiet=False,                   # Suppress output to avoid terminal flooding
            filter_mode="DLAs",           
            dla_phys_thresh="auto"        
        )
        
        # Trigger computation and store the batch results in memory
        self.tau_batch = self.ss.get_tau("H", 1, 1215)
        self.col_den_batch = self.ss.get_col_density("H", 1)
        
        # Clean up the temporary save file immediately
        if os.path.exists(save_file): 
            os.remove(save_file)

    def get_centered_spectrum(self, index=0):
        """Kinematically centers a specific spectrum from the batch to the Halo rest frame."""
        if self.tau_batch is None: 
            raise RuntimeError("Run generate_skewer_batch() first before fetching spectra.")

        nbins, vmax = self.ss.nbins, self.ss.vmax
        box_size = self.halo['box_size']
        a_scale = self.halo['scale_factor']
        
        # Extract target skewer data from the batch matrices
        tau_skewer = self.tau_batch[index, :]
        col_den_raw = self.col_den_batch[index, :]
        axis_skewer = self.ss.axis[index]
        
        flux_raw = np.exp(-tau_skewer)

        # Retrieve the axis of integration (0 for X, 1 for Y, 2 for Z)
        axis_idx = axis_skewer - 1 
        
        # Compute kinematic components along the line of sight for the target halo
        v_cosmo = (self.halo['cm'][axis_idx] / box_size) * vmax
        v_pec = self.halo['vel_raw'][axis_idx] * np.sqrt(a_scale)
        v_sys = v_cosmo + v_pec

        v_axis = np.arange(nbins) * (vmax / nbins)
        v_rel = v_axis - v_sys
        
        # Periodic wrapping for velocities
        v_rel_wrapped = (v_rel + vmax / 2.0) % vmax - vmax / 2.0
        
        sort_idx = np.argsort(v_rel_wrapped)
        max_idx = min(len(flux_raw), len(v_rel_wrapped))
        sort_idx = sort_idx[sort_idx < max_idx]

        # Standardize grid
        v_common = np.linspace(-vmax / 2.0, vmax / 2.0, nbins)
        flux_centered = np.interp(v_common, v_rel_wrapped[sort_idx], flux_raw[sort_idx])
        col_den_centered = np.interp(v_common, v_rel_wrapped[sort_idx], col_den_raw[sort_idx])

        wave_common = self.lam_rest * np.exp(v_common / self.c_light)
        chi_v_common = v_common / (vmax / box_size)

        return wave_common, v_common, chi_v_common, flux_centered, col_den_centered, self.lam_rest

    def get_total_col_den(self, index=0):
        """Returns the total Column Density (N_HI) for a specific skewer in the batch."""
        if self.col_den_batch is None: 
            return 0.0
        return np.sum(self.col_den_batch[index, :])