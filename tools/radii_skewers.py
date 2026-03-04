# In tools/radii_skewers.py

import numpy as np
import h5py
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
try:
    # Try direct import (if running in the tools folder)
    import utils
except ImportError:
    # If running from notebooks or the root folder
    from tools import utils

# Log Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - RADIISK - %(levelname)s - %(message)s')

class SkewerGeometry:
    def __init__(self, sim_context, halo_info):
        self.ctx = sim_context
        self.halo = halo_info
        self.box = halo_info['box_size']
        
        # Use R200 provided by HaloFinder (utils.py), with fallback
        self.r200_mpc = halo_info.get('r200', 0.2) 
        if self.r200_mpc <= 0: 
            self.r200_mpc = 0.2

    def _pbc(self, val):
        """Applies Periodic Boundary Conditions."""
        if val < 0: return val + self.box
        if val >= self.box: return val - self.box
        return val

    def generate_12_skewers(self, r_multiplier):
        """
        Generates 12 skewers tangent to an impact radius R = multiplier * R200.
        Axes are 0 (X), 1 (Y), 2 (Z) for internal logic.
        """
        impact_mpc = r_multiplier * self.r200_mpc
        
        skewers = []
        
        # Definition of perpendicular offsets
        offsets = [
            (impact_mpc, 0, "Right"),
            (-impact_mpc, 0, "Left"),
            (0, impact_mpc, "Top"),
            (0, -impact_mpc, "Bottom")
        ]
        
        sk_id = 0
        
        # --- AXIS 0 (X) - 4 Skewers ---
        for off_y, off_z, tag in offsets:
            sk_id += 1
            skewers.append({
                'id': sk_id, 'axis': 0, 'axis_name': 'X',
                'impact_mpc': impact_mpc, 'r_mult': r_multiplier,
                'offset_vec': np.array([0, off_y, off_z]), 
                'desc': f"Run X | Y={off_y:.3f}, Z={off_z:.3f} (Rel)"
            })
            
        # --- AXIS 1 (Y) - 4 Skewers ---
        for off_x, off_z, tag in offsets:
            sk_id += 1
            skewers.append({
                'id': sk_id, 'axis': 1, 'axis_name': 'Y',
                'impact_mpc': impact_mpc, 'r_mult': r_multiplier,
                'offset_vec': np.array([off_x, 0, off_z]),
                'desc': f"Run Y | X={off_x:.3f}, Z={off_z:.3f} (Rel)"
            })
            
        # --- AXIS 2 (Z) - 4 Skewers ---
        for off_x, off_y, tag in offsets:
            sk_id += 1
            skewers.append({
                'id': sk_id, 'axis': 2, 'axis_name': 'Z',
                'impact_mpc': impact_mpc, 'r_mult': r_multiplier,
                'offset_vec': np.array([off_x, off_y, 0]),
                'desc': f"Run Z | X={off_x:.3f}, Y={off_y:.3f} (Rel)"
            })
            
        return skewers

    def run_analysis(self, skewers_list, center_mode='gas'):
        """
        Runs the fake_spectra analysis and collects metadata + moments.
        Calculates both Flux-Weighted Velocity and HI-Coherent Velocity.
        Isolation is strictly fixed to the subbox edges (L/2) to avoid PBC wrap-around.
        """
        results = []
        base_cm = self.halo['cm'].copy()
        
        # The isolation radius is strictly the distance from the center to the edge
        iso_radius_mpc = self.box / 2.0 

        for sk in skewers_list:
            h_tmp = self.halo.copy()
            new_cm = base_cm + sk['offset_vec']
            
            # PBC Check
            for k in range(3):
                if new_cm[k] < 0: new_cm[k] += self.box
                if new_cm[k] >= self.box: new_cm[k] -= self.box
            
            h_tmp['cm'] = new_cm

            eng = utils.SpectrumEngine(self.ctx, h_tmp)
            
            try:
                # 1. Generate full skewer
                # Important: FakeSpectra expects axis 1,2,3. We store 0,1,2.
                # So we pass sk['axis'] + 1
                eng.generate_skewer(axis=sk['axis'] + 1)
                
                # 2. Get Spectrum (Calculates N_HI internally)
                w, f, l = eng.get_centered_spectrum(
                    mode=center_mode, 
                    isolation_radius_mpc=iso_radius_mpc
                )
                
                res = sk.copy()
                
                # --- Spectral Data ---
                res['wave'] = w
                res['flux'] = f
                
                # --- CALCULATION OF MOMENTS ---
                
                # A. Equivalent Width (Moment 0)
                absorption = 1.0 - f
                ew = np.trapz(absorption, x=w)
                res['eq_width'] = ew

                # Constants for velocity conversion
                c_light = 299792.458 # km/s
                lambda_rest = 1215.67

                # B. Flux-Weighted Mean Velocity
                # This tracks the ABSORPTION centroid (shadow)
                if ew > 0.001:
                    integrand_mom_1 = w * absorption
                    mean_wave = np.trapz(integrand_mom_1, x=w) / ew
                    
                    # Convert Mean Wavelength to Mean Velocity (km/s)
                    mean_vel = c_light * (mean_wave - lambda_rest) / lambda_rest
                    
                    # Higher Moments (Width/Skew/Kurt) based on Flux
                    delta_lambda = w - mean_wave
                    var = np.trapz((delta_lambda**2) * absorption, x=w) / ew
                    std_dev = np.sqrt(var)
                    
                    m3 = np.trapz((delta_lambda**3) * absorption, x=w) / ew
                    skewness = m3 / (std_dev**3) if std_dev > 0 else 0
                    
                    m4 = np.trapz((delta_lambda**4) * absorption, x=w) / ew
                    kurtosis = m4 / (std_dev**4) if std_dev > 0 else 0
                    
                else:
                    mean_wave = lambda_rest
                    mean_vel = 0.0
                    std_dev = 0.0
                    skewness = 0.0
                    kurtosis = 0.0

                res['mean_wave'] = mean_wave
                res['mean_vel'] = mean_vel # Flux-weighted
                res['std_dev'] = std_dev
                res['skewness'] = skewness
                res['kurtosis'] = kurtosis

                # C. Coherent HI Velocity
                # This tracks the GAS MASS centroid.
                # Formula: Sum(N_HI * v) / Sum(N_HI)
                
                if eng.col_den_centered is not None and np.sum(eng.col_den_centered) > 0:
                    # Convert wavelength array to velocity array relative to halo center
                    # v = c * (lambda - lambda_0) / lambda_0
                    vel_axis = c_light * (w - lambda_rest) / lambda_rest
                    
                    # Calculate Density-Weighted Mean Velocity
                    n_hi = eng.col_den_centered
                    
                    # Integration (Use trapz for consistency with grid)
                    coherent_vel = np.trapz(n_hi * vel_axis, x=vel_axis) / np.trapz(n_hi, x=vel_axis)
                    
                    res['vel_coherent'] = coherent_vel
                else:
                    res['vel_coherent'] = 0.0

                # --- Metadata ---
                res['halo_id'] = self.halo['id']
                res['halo_mass'] = self.halo['mass']
                res['halo_r200'] = self.r200_mpc 
                res['redshift'] = self.halo['redshift']
                res['impact_kpc'] = sk['impact_mpc'] * 1000 
                res['n_hi_col'] = eng.n_hi_masked 
                
                if eng.n_hi_masked > 0:
                    res['log_n_hi'] = np.log10(eng.n_hi_masked)
                else:
                    res['log_n_hi'] = 0.0

                results.append(res)
            except Exception as e:
                logging.error(f"Fail skewer {sk['id']}: {e}")
        
        return results