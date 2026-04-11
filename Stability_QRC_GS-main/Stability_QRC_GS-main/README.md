# Stability analysis with quantum reservoir computers and robust design with generalized synchronization

This repository provides a framework for studying the stability of chaotic dynamical systems using **Quantum Reservoir Computing (QRC) and Recurrence-free Quantum reservoir computing (RF-QRC)**. We also use **Generalised Synchronisation (GS) theory** to design robust quantum reservoir computers. It includes tools for computing invariant properties such as **Lyapunov exponents**, **Covariant Lyapunov Vectors (CLVs)** and **Conditional Lyapunov Exponents (CLEs)** via a Qiskit-based implementation.
## Repository Structure

- **`src/Notebook.ipynb`**  
  Contains the main workflow to evaluate stability properties with quantum reservoir computers. Input data can be modified to extend the framework to other systems such as **Lorenz96** for which the solvers are already added. Quantum noise models and the number of measurement **shots** can also be varied.

- **`src/QRC/qrc.py`**  
  Implements the **Qiskit-based quantum reservoir** and the methods required to compute **Lyapunov exponents**, including both conditional and autonomous versions, using quantum circuit simulations.

## Usage

To get started:
First, install the required dependencies:
```bash
pip install -r requirements.txt
```
Then run the following notebook
```bash

python Notebook.ipynb
```

## Citation
If you use this code in your research, please cite the corresponding paper:

Robust quantum reservoir computers for forecasting chaotic dynamics: generalized synchronization and stability (https://doi.org/10.1098/rspa.2025.0550)


