# Code Structure

This repository contains tools for FEFF (Finite Difference Method Near-Edge Structure) calculations and X-ray absorption spectroscopy analysis.

# FEFF10 Installation Guide

This guide provides instructions to install the **Intel Fortran compiler (`ifort`)** and build **FEFF10**, a powerful code for ab initio multiple scattering calculations.

---

## 📦 1. Install Intel Fortran Compiler (`ifort`)

1. Download the standalone `ifort` compiler from the [Intel oneAPI Standalone Components page](https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#fortran).

   > 💡 Tip: For Linux systems, you may want to download the larger package that includes all required dependencies.

2. Once installed, set up the environment by sourcing the setup script:
   ```bash
   source ~/intel/oneapi/setvars.sh
   ```

---

## ⚙️ 2. Build FEFF10

### Clone the repository

```bash
git clone https://github.com/times-software/feff10
cd feff10/src
```

### Choose a Fortran Compiler

FEFF10 supports both `ifort` and `gfortran`. To switch to `gfortran`:

- Open `compiler.mk` in a text editor.
- Scroll to the bottom and uncomment the `gfortran` configuration block.

### Compile FEFF10

#### Standard Build

```bash
make clean
make
```

#### MPI Build

```bash
make clean
make mpi
```

---

## ✅ Done!

After compilation, the FEFF executables should be ready to use. 🎉

---

## 🧠 Notes

- Make sure your system has MPI installed if you plan to use the MPI version.
- If you encounter any issues, double-check that the Fortran compiler is correctly set in `compiler.mk`.



## Directory Structure

```
├── src/
│   └── FEFF.py                 # Main FEFF computation module
|   └── submit_job.sh           # Slurm job submission template
├── Extract_FEFF.ipynb          # Jupyter notebook for extraction of cross-section
└── FEFF_analysis.py            # Extra analysis of the FEFF calculations
```

## Module Overview

### 1. src/FEFF.py
**Purpose**: Core FEFF computation and cluster management
- **Cluster Submission**: Scripts for submitting FEFF calculations to high-performance computing clusters
- **Job Management**: Utilities for monitoring and managing cluster jobs
- **Data Processing**: Post-calculation data handling and organization

### 2. Extract_FEFF.ipynb
**Purpose**: Interactive data cross-section extraction
- **Compton Cross-Section Calculation**: Extracts and computes Compton scattering cross-sections from FEFF output
- **Data Visualization**: Interactive plots and analysis of spectroscopic data
- **Export Functions**: Saves processed results for further analysis

### 3. FEFF_analysis.py
**Purpose**: Automated analysis utilities
- **Data Persistence**: Functions to save analysis results to disk
- **Batch Processing**: Tools for analyzing multiple FEFF calculations

## Quick Start

1. **Run FEFF Calculations**: Use `src/FEFF.py` to submit jobs to your cluster
2. **Extract Results**: Open `Extract_FEFF.ipynb` for interactive data analysis
3. **Automated Analysis**: Use `FEFF_analysis.py` for batch processing and automated workflows

## Dependencies

- Python 3.x
- Jupyter Notebook
- NumPy, SciPy, Numba, ROOT (for numerical calculations)
- Matplotlib (for visualization)
- FEFF software package

## Usage Notes

- Ensure FEFF is properly installed and configured on your system
- Cluster submission scripts may need modification based on your HPC environment
- Review and adjust file paths in the analysis scripts for your directory structure