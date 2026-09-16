"""Data loading utilities following EffectiveConPerturb project standards"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Project constants following your existing patterns
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATAPATH = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def load_metadata(
    datapath: str = DATAPATH,
    metadata_file: str = None,
    columns: List[str] = None,
    sc_root: str = None
) -> pd.DataFrame:
    """Load subject IDs from SC matrix CSV filenames.
    
    Parameters
    ----------
    datapath : str
        Base data directory.
    metadata_file : str, optional
        Deprecated, kept for backward compatibility. Ignored.
    columns : List[str], optional
        Deprecated, kept for backward compatibility. Ignored.
    sc_root : str, optional
        SC folder name under data/SCs/ (should come from config file).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'IPP' column containing subject IDs from CSV basenames.
    """
    if sc_root is None:
        raise ValueError("sc_root must be provided (should come from config)")
    
    sc_dir = os.path.join(datapath, "SCs", sc_root)
    
    if not os.path.exists(sc_dir):
        raise FileNotFoundError(f"SC directory not found: {sc_dir}")
    
    subject_ids = set()
    
    # Check if sc_root has vendor subfolders (like workbench_siemens, workbench_GE)
    potential_vendors = [item for item in os.listdir(sc_dir) 
                        if os.path.isdir(os.path.join(sc_dir, item))]
    
    has_csv_files_in_root = any(f.endswith('.csv') for f in os.listdir(sc_dir) 
                               if os.path.isfile(os.path.join(sc_dir, f)))
    
    if has_csv_files_in_root:
        # CSV files are directly in sc_root
        for file in os.listdir(sc_dir):
            if file.endswith('.csv'):
                subject_id = file[:-4]  # Remove .csv extension
                subject_ids.add(subject_id)
    else:
        # Look for CSV files in vendor subfolders
        for vendor in potential_vendors:
            vendor_dir = os.path.join(sc_dir, vendor)
            if os.path.exists(vendor_dir):
                for file in os.listdir(vendor_dir):
                    if file.endswith('.csv'):
                        subject_id = file[:-4]  # Remove .csv extension
                        subject_ids.add(subject_id)
    
    if not subject_ids:
        raise FileNotFoundError(f"No CSV files found in {sc_dir} or its subdirectories")
    
    # Create DataFrame with IPP column (kept for backward compatibility)
    subject_list = sorted(list(subject_ids))
    return pd.DataFrame({"IPP": subject_list})

def load_sc_matrix(
    ipp: str,
    datapath: str = DATAPATH,
    sc_root: str = None,
    normalize: bool = True,
    threshold: float = 0.0,
    in_counts: bool = False
) -> np.ndarray:
    """Load SC matrix from CSV file.
    
    Parameters
    ----------
    ipp : str
        Subject identifier (CSV basename without extension).
    datapath : str
        Base data directory.
    sc_root : str
        SC folder name under data/SCs/ (should come from config file).
    normalize : bool
        Whether to apply normalization (scale by 0.2).
    threshold : float
        Threshold to zero out weak connections.
    in_counts : bool
        Whether matrix is in raw counts (skips initial max normalization).
    """
    if sc_root is None:
        raise ValueError("sc_root must be provided (should come from config)")
    
    sc_dir = os.path.join(datapath, "SCs", sc_root)
    
    # First check if CSV is directly in sc_root
    direct_path = os.path.join(sc_dir, f"{ipp}.csv")
    if os.path.exists(direct_path):
        mat = pd.read_csv(direct_path, header=None).values
        if not in_counts:
            mat = mat / max(mat.max(), 1e-6)  
        if threshold > 0:
            mat[mat < threshold] = 0  # Take out streamline threshold percentage of weakest links
        if normalize:    
            mat = mat * 0.2
        return mat
    
    # If not found directly, look in vendor subfolders
    if os.path.exists(sc_dir):
        for vendor in os.listdir(sc_dir):
            vendor_path = os.path.join(sc_dir, vendor)
            if os.path.isdir(vendor_path):
                csv_path = os.path.join(vendor_path, f"{ipp}.csv")
                if os.path.exists(csv_path):
                    mat = pd.read_csv(csv_path, header=None).values
                    if not in_counts:
                        mat = mat / max(mat.max(), 1e-6)
                    if threshold > 0:
                        mat[mat < threshold] = 0
                    if normalize:
                        mat = mat * 0.2
                    return mat

    raise FileNotFoundError(f"SC matrix for {ipp} not found in {sc_dir} or its subdirectories")

def load_all_sc_matrices(
    ipp_list: List[str],
    datapath: str = DATAPATH,
    sc_root: str = None,
    normalize: bool = True,
    threshold: float = 0.0,
    in_counts: bool = False
) -> Dict[str, np.ndarray]:
    """Load all SC matrices for given subject list.

    Parameters
    ----------
    ipp_list : List[str]
        Subject identifiers to load (CSV basenames).
    datapath : str
        Base data directory.
    sc_root : str
        SC folder name under data/SCs/ (should come from config file).
    normalize : bool
        Whether to apply normalization.
    threshold : float
        Threshold to zero out weak connections.
    in_counts : bool
        Whether matrices are in raw counts.
    """
    if sc_root is None:
        raise ValueError("sc_root must be provided (should come from config)")
        
    matrices: Dict[str, np.ndarray] = {}
    for ipp in ipp_list:
        try:
            matrices[ipp] = load_sc_matrix(ipp, datapath=datapath, sc_root=sc_root, normalize=normalize, threshold=threshold, in_counts=in_counts)
        except FileNotFoundError:
            pass  # Skip missing subjects
    return matrices
