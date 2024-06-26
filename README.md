# learning_circuit
Learning Linear Circuits

## Setup

0. Preliminaries for macOS: Install miniforge
   - Download https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
   - Run the script
   - Open another shell and run
     ```bash
     bash Miniforge3-MacOSX-arm64.sh
     ```

1. Clone a local copy of the repository:

```bash
git clone git@github.com:maguzj/learning_circuit.git
```

2. Create the environment. Follow one of these two options depending on your computer.
   - **For Intel chips:** The main directory contains an environment.yml file for easily setting up a conda environment, named cl, with all the package dependencies:
     ```bash
     conda env create --file=environment-intel.yml
     ```
     To activate the environment, run
     ```bash
     conda activate cl
     ```
   - **For M1/M2 chips:** We have to build numpy with the accelerator. (:warning: this is constantly evolving. New OS will get rid of these tricks. If you encounter any problem, please create an issue.)
     ```bash
     conda env create --file=environment-M1-M2.yml
     ```
     Activate the environment, install numpy using pip and set the pip to be recognized by further package installations:
     ```bash
     conda activate cl
     ```
     ```bash
     pip install --no-binary :all: numpy==1.24.3 --no-cache-dir
     ```
     Once numpy installed, run
     ```bash
     conda config --set pip_interop_enabled true
     ```

     check that numpy is using vecLib:
     ```bash
     >>> import numpy
     >>> numpy.show_config()
     ```
     If everything is right, you should see info like ```/System/Library/Frameworks/vecLib.framework/Headers``` printed.
     Then install the higher level dependencies.
     ```bash
     pip install -r requirements.txt --no-cache-dir
     ```

     (for more information see: https://gist.github.com/MarkDana/a9481b8134cf38a556cf23e1e815dafb)


3. Several jupyter notebooks are provided for getting started.