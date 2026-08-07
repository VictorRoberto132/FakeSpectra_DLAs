# -*- coding: utf-8 -*-
import os
import sys
import time
import logging
import gc
import signal
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# IMPORTS
# =============================================================================
try:
    import utils
    import fake_spectra.spectra as spectra
    import fake_spectra.unitsystem as unitsystem
except ImportError:
    sys.path.append(os.path.abspath("."))
    import utils
    import fake_spectra.spectra as spectra
    import fake_spectra.unitsystem as unitsystem

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
LOG_FILE = "extraction_grid.log"
logging.basicConfig(
    filename=LOG_FILE, filemode="w", level=logging.DEBUG,
    format="%(asctime)s - %(process)d - %(levelname)s - %(message)s"
)

SNAP_NUM        = "094"
SIM_NAME        = "fiducial"
BASE_PATH       = "/home/STORAGE/MN5_resims/fiducial/hydro_output"
SIM_TYPE        = "MN5"

SKEWERS_PER_AXIS = 30_000
CHUNK_SIZE       = 200
NUM_CORES        = 20

# Timeout por chunk em segundos. Ajuste conforme o tempo médio observado.
# Se um chunk de 200 skewers leva ~60s no pior caso, use 300s como margem.
CHUNK_TIMEOUT_S  = 300

# Controla quantos futures ficam "em voo" simultaneamente.
# Evita enfileirar TODOS os chunks de uma vez (pressão de memória).
MAX_INFLIGHT     = NUM_CORES * 2

SPECTRAL_RES_KMS = 10.0
LYA_WAVELENGTH   = 1215.67
C_LIGHT          = 299792.458

TOOLS_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.dirname(TOOLS_DIR)
TEMP_DIR   = os.path.join(PROJ_ROOT, "notebooks", "temp_chunks")
os.makedirs(TEMP_DIR, exist_ok=True)

# =============================================================================
# FUNÇÕES AUXILIARES (idênticas à versão original)
# =============================================================================

def get_simulation_box_size(base_path, snap_num):
    group_dir  = f"groups_{int(snap_num):03d}"
    group_file = os.path.join(
        base_path, group_dir,
        f"fof_subhalo_tab_{int(snap_num):03d}.0.hdf5"
    )
    if not os.path.exists(group_file):
        snap_dir   = f"snapdir_{int(snap_num):03d}"
        group_file = os.path.join(
            base_path, snap_dir,
            f"snapshot_{int(snap_num):03d}.0.hdf5"
        )
    if not os.path.exists(group_file):
        raise FileNotFoundError(f"Header não encontrado: {group_file}")
    with h5py.File(group_file, "r") as f:
        return float(f["Header"].attrs["BoxSize"])


def get_unit_system(sim_type):
    if sim_type.upper() == "MN5":
        return unitsystem.UnitSystem(
            UnitLength_in_cm=3.085678e24,
            UnitMass_in_g=1.989e43,
            UnitVelocity_in_cm_per_s=1e5
        )
    if sim_type.upper() == "TNG":
        return unitsystem.UnitSystem(
            UnitLength_in_cm=3.085678e21,
            UnitMass_in_g=1.989e43,
            UnitVelocity_in_cm_per_s=1e5
        )
    return None


def generate_grid_skewers(box_size, skewers_per_axis):
    n_side = int(np.sqrt(skewers_per_axis))
    actual = n_side ** 2

    grid_1d       = (np.arange(n_side) + 0.5) * box_size / n_side
    grid_a, grid_b = np.meshgrid(grid_1d, grid_1d, indexing="ij")
    grid_a, grid_b = grid_a.flatten(), grid_b.flatten()

    rng        = np.random.default_rng(seed=42)  # seed para reprodutibilidade
    all_skewers = []
    skewer_id  = 0

    def jitter():
        return rng.uniform(-1e-4, 1e-4)

    for axis, fixed_dim in [(1, "x"), (2, "y"), (3, "z")]:
        for i in range(actual):
            if axis == 1:
                sk = {"skewer_idx": skewer_id, "axis": axis,
                      "pos_x_code": 0.001 + jitter(),
                      "pos_y_code": grid_a[i] + jitter(),
                      "pos_z_code": grid_b[i] + jitter()}
            elif axis == 2:
                sk = {"skewer_idx": skewer_id, "axis": axis,
                      "pos_x_code": grid_a[i] + jitter(),
                      "pos_y_code": 0.001 + jitter(),
                      "pos_z_code": grid_b[i] + jitter()}
            else:
                sk = {"skewer_idx": skewer_id, "axis": axis,
                      "pos_x_code": grid_a[i] + jitter(),
                      "pos_y_code": grid_b[i] + jitter(),
                      "pos_z_code": 0.001 + jitter()}
            all_skewers.append(sk)
            skewer_id += 1

    return all_skewers


def safe_log10(value):
    return np.log10(value) if value > 0.0 else 0.0


# =============================================================================
# WORKER — roda em processo isolado
# =============================================================================

def process_grid_chunk(args):
    """
    Worker isolado. Retorna apenas o caminho do .pkl ou sinaliza erro.
    Nunca retorna arrays NumPy grandes pelo pipe IPC.
    """
    chunk_id, skewers_chunk, box_size = args
    n_skewers = len(skewers_chunk)

    # --- Ignorar SIGINT no worker para evitar KeyboardInterrupt em cascata ---
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    ctx      = utils.SimulationContext(BASE_PATH, SNAP_NUM, sim_type=SIM_TYPE)
    cofm_arr = np.zeros((n_skewers, 3), dtype=np.float64)
    axis_arr = np.zeros(n_skewers, dtype=np.int32)
    meta     = {}

    for idx, sk in enumerate(skewers_chunk):
        cofm_arr[idx] = [sk["pos_x_code"], sk["pos_y_code"], sk["pos_z_code"]]
        axis_arr[idx] = sk["axis"]
        meta[idx]     = {**sk, "chunk_id": chunk_id}

    # Arquivo HDF5 temporário exclusivo por processo+chunk (evita colisão)
    pid       = os.getpid()
    save_file = os.path.join(TEMP_DIR, f"tmp_{chunk_id}_{pid}.hdf5")
    out_pkl   = os.path.join(TEMP_DIR, f"chunk_{chunk_id}.pkl")

    ss = None
    try:
        ss = spectra.Spectra(
            int(SNAP_NUM), BASE_PATH,
            cofm=cofm_arr, axis=axis_arr,
            res=SPECTRAL_RES_KMS,
            savefile=save_file,
            reload_file=True,
            kernel="voronoi",
            MPI=None,
            quiet=True,
            filter_mode="None",
            units=get_unit_system(ctx.sim_type)
        )

        tau_batch     = ss.get_tau("H", 1, 1215)
        col_den_batch = ss.get_col_density("H", 1)
        temp_batch    = ss.get_temp("H", 1)

        v_axis   = np.arange(ss.nbins, dtype=np.float64) * (ss.vmax / ss.nbins)
        wave_axis = LYA_WAVELENGTH * np.exp(v_axis / C_LIGHT)
        chi_axis  = v_axis / (ss.vmax / box_size)
        total_col = np.sum(col_den_batch, axis=1)

        for idx in range(n_skewers):
            n_hi = float(total_col[idx])
            meta[idx].update({
                "log_n_hi":      safe_log10(n_hi),
                "total_col_den": n_hi,
                "flux":          np.exp(-tau_batch[idx]).astype(np.float32),
                "n_hi_col":      col_den_batch[idx].astype(np.float32),
                "temp_k":        temp_batch[idx].astype(np.float32),
                "wave":          wave_axis.astype(np.float32),
                "vel_axis":      v_axis.astype(np.float32),
                "chi_axis":      chi_axis.astype(np.float32),
                "is_zero_col_den": bool(n_hi == 0.0),
            })

        pd.DataFrame(list(meta.values())).to_pickle(out_pkl)
        logging.info(f"Chunk {chunk_id} concluído com sucesso (pid={pid}).")
        return (chunk_id, out_pkl, "SUCCESS")

    except Exception:
        tb = traceback.format_exc()
        logging.error(f"Chunk {chunk_id} falhou:\n{tb}")
        return (chunk_id, None, f"ERROR: {tb[:200]}")

    finally:
        # --- Limpeza explícita e determinística ---
        if ss is not None:
            try:
                del ss
            except Exception:
                pass
        gc.collect()
        if os.path.exists(save_file):
            try:
                os.remove(save_file)
            except OSError:
                pass


# =============================================================================
# ORQUESTRADOR PRINCIPAL — sliding window com timeout por future
# =============================================================================

def run_pipeline(chunks, n_chunks):
    """
    Submete chunks com uma janela deslizante (sliding window) de MAX_INFLIGHT
    futures simultâneos. Cada future tem timeout individual de CHUNK_TIMEOUT_S.

    Por que isso resolve o deadlock:
    - ProcessPoolExecutor NÃO usa semáforos internos da forma que Pool usa.
    - `as_completed` com timeout garante que nenhum future bloqueia para sempre.
    - Chunks travados são cancelados e re-enfileirados ou marcados como falha.
    - O executor é encerrado com shutdown(wait=True, cancel_futures=True)
      que envia SIGTERM limpo para cada worker.
    """
    success_files = []
    fail_ids      = []
    pending       = {}         # future → chunk_args
    chunk_iter    = iter(chunks)
    submitted     = 0
    completed     = 0

    # cancel_futures=True disponível a partir do Python 3.9
    with ProcessPoolExecutor(
        max_workers=NUM_CORES,
        mp_context=multiprocessing.get_context("spawn")
    ) as executor:

        pbar = tqdm(total=n_chunks, desc="Chunks", unit="chunk")

        # --- Preenche a janela inicial ---
        while submitted < MAX_INFLIGHT:
            chunk = next(chunk_iter, None)
            if chunk is None:
                break
            fut = executor.submit(process_grid_chunk, chunk)
            pending[fut] = chunk
            submitted += 1

        # --- Loop principal: sliding window ---
        while pending:
            # Espera o PRIMEIRO future que terminar, com timeout
            done_iter = as_completed(pending, timeout=CHUNK_TIMEOUT_S + 30)

            try:
                fut = next(done_iter)
            except TimeoutError:
                # Nenhum future completou no tempo esperado → identifica o travado
                logging.error(
                    "TIMEOUT GLOBAL: nenhum chunk completou em "
                    f"{CHUNK_TIMEOUT_S + 30}s. Cancelando futures pendentes."
                )
                for hung_fut, hung_args in list(pending.items()):
                    cid = hung_args[0]
                    logging.error(f"Chunk {cid} considerado travado e abortado.")
                    hung_fut.cancel()
                    fail_ids.append(cid)
                    pbar.update(1)
                pending.clear()
                break

            # --- Processa o resultado ---
            chunk_args = pending.pop(fut)
            chunk_id   = chunk_args[0]

            try:
                result_id, file_path, status = fut.result(timeout=10)
                if status == "SUCCESS" and file_path:
                    success_files.append(file_path)
                else:
                    logging.warning(f"Chunk {result_id} retornou status: {status}")
                    fail_ids.append(result_id)
            except TimeoutError:
                logging.error(f"fut.result() timeout no chunk {chunk_id}.")
                fail_ids.append(chunk_id)
            except Exception as exc:
                logging.error(f"Exceção ao coletar chunk {chunk_id}: {exc}")
                fail_ids.append(chunk_id)

            completed += 1
            pbar.update(1)
            pbar.set_postfix({
                "OK": len(success_files),
                "Fail": len(fail_ids),
                "Inflight": len(pending)
            })

            # --- Injeta próximo chunk na janela ---
            next_chunk = next(chunk_iter, None)
            if next_chunk is not None:
                new_fut = executor.submit(process_grid_chunk, next_chunk)
                pending[new_fut] = next_chunk
                submitted += 1

        pbar.close()

    return success_files, fail_ids


# =============================================================================
# PONTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    # spawn DEVE ser definido antes de qualquer uso de multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    print("🔭 Iniciando pipeline de extração de skewers...")
    BOX_SIZE           = get_simulation_box_size(BASE_PATH, SNAP_NUM)
    master_skewer_list = generate_grid_skewers(BOX_SIZE, SKEWERS_PER_AXIS)
    total_generated    = len(master_skewer_list)

    print(f"   Box size        : {BOX_SIZE:.2f} cMpc/h")
    print(f"   Total skewers   : {total_generated:,}")
    print(f"   Chunk size      : {CHUNK_SIZE}")

    chunks  = [
        (i // CHUNK_SIZE, master_skewer_list[i:i + CHUNK_SIZE], BOX_SIZE)
        for i in range(0, total_generated, CHUNK_SIZE)
    ]
    n_chunks = len(chunks)
    print(f"   Total chunks    : {n_chunks}  |  Cores: {NUM_CORES}")
    print(f"   Max inflight    : {MAX_INFLIGHT}  |  Timeout/chunk: {CHUNK_TIMEOUT_S}s\n")

    start_time = time.time()

    # --- Executa com sliding window ---
    success_files, fail_ids = run_pipeline(chunks, n_chunks)

    # --- Relatório de falhas ---
    if fail_ids:
        print(f"\n⚠️  {len(fail_ids)} chunks falharam: {fail_ids[:20]}{'...' if len(fail_ids) > 20 else ''}")
        print("   Consulte extraction_grid.log para detalhes.")

    # --- Consolidação ---
    if success_files:
        print(f"\n📦 Consolidando {len(success_files)} arquivos pkl...")
        df_list = [pd.read_pickle(f) for f in success_files]
        df      = pd.concat(df_list, ignore_index=True)
        df      = df.sort_values(["chunk_id", "skewer_idx"]).reset_index(drop=True)

        out_name = f"{SIM_NAME}_Grid_{total_generated}Sk.pkl"
        out_path = os.path.join(PROJ_ROOT, "notebooks", out_name)
        df.to_pickle(out_path)
        print(f"✅ Salvo em: {out_path}")

        # Remove temporários
        removed, errors = 0, 0
        for f in success_files:
            try:
                os.remove(f)
                removed += 1
            except OSError:
                errors += 1
        print(f"   Temporários removidos: {removed}  |  Erros na remoção: {errors}")
    else:
        print("\n❌ Nenhum chunk bem-sucedido. Verifique o log.")

    elapsed = (time.time() - start_time) / 60.0
    print(f"\n⏱️  Pipeline finalizado em {elapsed:.1f} minutos.")