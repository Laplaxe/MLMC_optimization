# Machine Learning for Monte Carlo

This directory contains a series of files to train and use machine learning architectures to aid the Monte Carlo sampling of the 2D Edwards-Anderson model.

The directory contains the following files:
- N4.py: file related to the N4 architecture (https://arxiv.org/pdf/2407.19483).
- data_loads.py: directory to load the data that are used for training. The data are assumed to be in a dati_PT directory suitably formatted (see below).
- ellMade.py: file related to the $\ell$-MADE architecture (https://arxiv.org/pdf/2407.19483)
- geometry.py: file to handle the geometry of the problem (e.g., the ordering of the spins in a spiral update).
- global_steps.py: global steps for the MADE and NADE architectures (e.g generation of the configurations)
- made.py: file related to the MADE architecture.
- nade.py: file related to the NADE architecture.
- utilities.py: some utility functions (e.g., computation of the best-spaced temperatures based on the specific heat).

## Organization of the training set (dati_PT)

The `dati_PT` directory contains simulation data organized into multiple subdirectories based on the system's linear size (`L`) and the simulation run number (`run`). Each subdirectory follows a consistent hierarchical structure, described as follows:

### Top-Level Directories
Each directory at this level is named in the format `runX_LY`, where:
- `X` is the **run number**.
- `Y` is the **linear size** of the system.

### Nested Structure
Inside each `runX_LY` directory:
- There is a `repeat/` folder containing subdirectories for different seeds.
- Each seed-specific subdirectory is named `LY_SZ`, where:
  - `Z` is the **seed number** used in the simulation.

### Seed Subdirectories
Each seed subdirectory contains:
- Text files named `temperature_<value>.txt`, which store simulation results for different temperature values.

### Example

For `run1_L8`:
```plaintext
dati_PT/
├── run1_L8/
│   ├── repeat/
│   │   ├── L8_S42/
│   │   │   ├── temperature_0.45.txt
│   │   │   ├── temperature_1.02.txt
│   │   │   ├── ...
│   │   ├── L8_S123456/
│   │   ├── L8_S010101/
│   │   ├── ...
