import numpy as np
import logging

try:
    import utils
except ImportError:
    from tools import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - RADIISK - %(levelname)s - %(message)s')

class SkewerGeometry:
    """Manages the spatial orientation of skewers around the halo using native Code Units."""
    def __init__(self, sim_context, halo_info):
        self.ctx = sim_context
        self.halo = halo_info
        self.box = halo_info['box_size']
        
        # Native Code Unit (Proven to be cMpc/h for MN5 via diagnostic)
        self.r200_code = halo_info.get('r200', 0.2) 
        if self.r200_code <= 0: 
            self.r200_code = 0.2

    def run_analysis_batched(self, skewers_list):
        """
        Processa múltiplos skewers simultaneamente. 
        Envia arrays de coordenadas (N, 3) e eixos (N,) para forçar
        o fake_spectra a ler os dados do disco apenas uma vez.
        """
        results = []
        base_cm = self.halo['cm'].copy()
        total_skewers = len(skewers_list)

        # 1. Agrupando todas as coordenadas e eixos
        cofm_list = []
        axis_list = []
        
        for sk in skewers_list:
            new_cm = base_cm + sk['offset_vec']
            
            # Apply Periodic Boundary Conditions (PBC) purely in Code Units
            for k in range(3):
                if new_cm[k] < 0: new_cm[k] += self.box
                if new_cm[k] >= self.box: new_cm[k] -= self.box
                
            cofm_list.append(new_cm)
            axis_list.append(sk['axis'] + 1) # fake_spectra usa 1, 2, 3 (X, Y, Z)

        # Vetorização nativa
        cofm_array = np.array(cofm_list, dtype=np.float64)
        axis_array = np.array(axis_list, dtype=np.int32)

        print(f"--> [Halo {self.halo['id']}] Extraindo lote de {total_skewers} Skewers em RAM...", flush=True)

        # 2. Instancia o motor e roda o lote inteiro de uma vez
        eng = utils.SpectrumEngine(self.ctx, self.halo)
        
        try:
            # Novo método que precisaremos criar no SpectrumEngine
            eng.generate_skewer_batch(axis_array, cofm_array)
            
            # 3. Resgata e calcula os momentos para cada skewer individualmente
            for idx, sk in enumerate(skewers_list):
                
                # Resgata os grids usando o índice do skewer no lote
                w, v_arr, chi_arr, f, n_hi_arr, lam_rest = eng.get_centered_spectrum(index=idx)
                
                res = sk.copy()
                res['wave'] = w
                res['vel_axis'] = v_arr
                res['chi_axis'] = chi_arr
                res['flux'] = f
                res['n_hi_col'] = n_hi_arr  
                
                # --- MOMENTS CALCULATION ---
                absorption = 1.0 - f
                ew = np.trapz(absorption, x=w)
                res['eq_width'] = ew
                c_light = 299792.458 

                if ew > 0.001:
                    norm_v = np.trapz(absorption, x=v_arr)
                    mean_vel = np.trapz(v_arr * absorption, x=v_arr) / norm_v
                    delta_v = v_arr - mean_vel
                    var = np.trapz((delta_v**2) * absorption, x=v_arr) / norm_v
                    std_dev = np.sqrt(var)
                    
                    m3 = np.trapz((delta_v**3) * absorption, x=v_arr) / norm_v
                    skewness = m3 / (std_dev**3) if std_dev > 0 else 0
                    
                    m4 = np.trapz((delta_v**4) * absorption, x=v_arr) / norm_v
                    kurtosis = m4 / (std_dev**4) if std_dev > 0 else 0
                    
                    mean_wave = lam_rest * np.exp(mean_vel / c_light)
                else:
                    mean_wave = lam_rest
                    mean_vel = std_dev = skewness = kurtosis = 0.0

                res['mean_wave'] = mean_wave
                res['mean_vel'] = mean_vel
                res['std_dev'] = std_dev
                res['skewness'] = skewness
                res['kurtosis'] = kurtosis

                if np.sum(n_hi_arr) > 0:
                    coherent_vel = np.trapz(n_hi_arr * v_arr, x=v_arr) / np.trapz(n_hi_arr, x=v_arr)
                    res['vel_coherent'] = coherent_vel
                else:
                    res['vel_coherent'] = 0.0

                # Metadata Consolidation
                res['halo_id'] = self.halo['id']
                res['halo_mass'] = self.halo['mass']
                res['halo_r200'] = self.halo['r200']
                res['redshift'] = self.halo['redshift']
                
                # Nota: a densidade de coluna total agora precisa ser pega por índice também
                col_den_total = eng.get_total_col_den(index=idx)
                res['total_col_den'] = col_den_total 
                res['log_n_hi'] = np.log10(col_den_total) if col_den_total > 0 else 0.0
                
                res['impact_kpc'] = sk['impact_code'] * 1000.0

                results.append(res)
                
        except Exception as e:
            logging.error(f"Falha ao processar o lote no Halo {self.halo['id']}: {e}")
    
        return results        

    def generate_skewers(self, r_multiplier):
        """Generates skewer coordinates. Returns 3 for r=0.0, and 12 for r>0.0."""
        impact_code = r_multiplier * self.r200_code
        skewers = []
        sk_id = 0

        if r_multiplier == 0.0:
            for ax, name in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
                sk_id += 1
                skewers.append({
                    'id': sk_id, 
                    'axis': ax, 
                    'axis_name': name, 
                    'impact_code': 0.0, 
                    'r_mult': 0.0, 
                    'offset_vec': np.array([0.0, 0.0, 0.0])
                })
            return skewers

        offsets = [
            (impact_code, 0, "Right"),
            (-impact_code, 0, "Left"),
            (0, impact_code, "Top"),
            (0, -impact_code, "Bottom")
        ]
        
        for off_y, off_z, tag in offsets:
            sk_id += 1
            skewers.append({'id': sk_id, 'axis': 0, 'axis_name': 'X', 'impact_code': impact_code, 'r_mult': r_multiplier, 'offset_vec': np.array([0, off_y, off_z])}) 
            
        for off_x, off_z, tag in offsets:
            sk_id += 1
            skewers.append({'id': sk_id, 'axis': 1, 'axis_name': 'Y', 'impact_code': impact_code, 'r_mult': r_multiplier, 'offset_vec': np.array([off_x, 0, off_z])})
            
        for off_x, off_y, tag in offsets:
            sk_id += 1
            skewers.append({'id': sk_id, 'axis': 2, 'axis_name': 'Z', 'impact_code': impact_code, 'r_mult': r_multiplier, 'offset_vec': np.array([off_x, off_y, 0])})
            
        return skewers

    def run_analysis(self, skewers_list):
        results = []
        base_cm = self.halo['cm'].copy()

        total_skewers = len(skewers_list)

        for idx, sk in enumerate(skewers_list):
            
            print(f"--> [Halo {self.halo['id']}] Extraindo Skewer {idx+1}/{total_skewers} | Impacto: {sk['r_mult']} R200 | Eixo: {sk['axis_name']}", flush=True)
            h_tmp = self.halo.copy()
            new_cm = base_cm + sk['offset_vec']
            
            # Apply Periodic Boundary Conditions (PBC) purely in Code Units
            for k in range(3):
                if new_cm[k] < 0: new_cm[k] += self.box
                if new_cm[k] >= self.box: new_cm[k] -= self.box
            
            h_tmp['cm'] = new_cm
            eng = utils.SpectrumEngine(self.ctx, h_tmp)
            
            try:
                # Run native extraction
                eng.generate_skewer(axis=sk['axis'] + 1)
                
                # Retrieve grids (No isolation radius applied)
                w, v_arr, chi_arr, f, n_hi_arr, lam_rest = eng.get_centered_spectrum()
                
                res = sk.copy()
                res['wave'] = w
                res['vel_axis'] = v_arr
                res['chi_axis'] = chi_arr
                res['flux'] = f
                res['n_hi_col'] = n_hi_arr  
                
                # --- MOMENTS CALCULATION ---
                absorption = 1.0 - f
                ew = np.trapz(absorption, x=w)
                res['eq_width'] = ew
                c_light = 299792.458 

                if ew > 0.001:
                    norm_v = np.trapz(absorption, x=v_arr)
                    mean_vel = np.trapz(v_arr * absorption, x=v_arr) / norm_v
                    delta_v = v_arr - mean_vel
                    var = np.trapz((delta_v**2) * absorption, x=v_arr) / norm_v
                    std_dev = np.sqrt(var)
                    
                    m3 = np.trapz((delta_v**3) * absorption, x=v_arr) / norm_v
                    skewness = m3 / (std_dev**3) if std_dev > 0 else 0
                    
                    m4 = np.trapz((delta_v**4) * absorption, x=v_arr) / norm_v
                    kurtosis = m4 / (std_dev**4) if std_dev > 0 else 0
                    
                    mean_wave = lam_rest * np.exp(mean_vel / c_light)
                else:
                    mean_wave = lam_rest
                    mean_vel = std_dev = skewness = kurtosis = 0.0

                res['mean_wave'] = mean_wave
                res['mean_vel'] = mean_vel
                res['std_dev'] = std_dev
                res['skewness'] = skewness
                res['kurtosis'] = kurtosis

                if np.sum(n_hi_arr) > 0:
                    coherent_vel = np.trapz(n_hi_arr * v_arr, x=v_arr) / np.trapz(n_hi_arr, x=v_arr)
                    res['vel_coherent'] = coherent_vel
                else:
                    res['vel_coherent'] = 0.0

                # Metadata Consolidation
                res['halo_id'] = self.halo['id']
                res['halo_mass'] = self.halo['mass']
                res['halo_r200'] = self.halo['r200']
                res['redshift'] = self.halo['redshift']
                res['total_col_den'] = eng.n_hi_masked 
                res['log_n_hi'] = np.log10(eng.n_hi_masked) if eng.n_hi_masked > 0 else 0.0
                
                # Conversion to kpc solely for human-readable dataframe logging
                res['impact_kpc'] = sk['impact_code'] * 1000.0

                results.append(res)
                
            except Exception as e:
                logging.error(f"Fail skewer {sk['id']} at Halo {self.halo['id']}: {e}")
        
        return results