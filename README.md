# Quantum Reservoir Computing - Applicability POC (Evaluation 1)

This project contains the implementation and experiments for studying the appliction of  **Quantum Reservoir Computing (QRC)** to Disruption Prediction and control in Nuclear Fusion. Done on top of the work: Robust quantum reservoir computers for forecasting chaotic dynamics: generalized synchronization and stability (https://doi.org/10.1098/rspa.2025.0550): 

## Project Structure

- `Stability_QRC_GS-main/`: The core source code for the QRC framework.
- `Noise_Robustness_Experiment/`: Comparing QRC and CRC under different noise levels.
- `Quantum_Adv/`: Comparing QRC and CRC for transition/tipping point detection.
- `VPT_Experiment/`: Measuring Valid Prediction Time (VPT) for both QRC and CRC.
- `requirements.txt`: List of dependencies required to run the project.


### Install Dependencies (For both the experiments and the core notebook)
- Suggested Python version: 3.10.x .
- It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

## Running the Experiments

All experiments should be run from the **root directory** of the project to ensure the `sys.path` additions in the scripts work correctly.

### 1. Noise Robustness Experiment
Compares the performance of Quantum Reservoir Computing (QRC) and Classical Reservoir Computing (CRC) under varying levels of Gaussian noise.
```bash
python Noise_Robustness_Experiment/noise_robustness.py
```
**Output:** `Noise_Robustness_Experiment/Noise_Robustness_QRC_vs_CRC.png`

### 2. Quantum Advantage / Transition Experiment
Analyzes the detection of tipping points and transitions in chaotic systems, comparing early warning signals between QRC and CRC.
```bash
python Quantum_Adv/transition_experiment.py
```
**Output:** `transition_results_comparison.png`

### 3. Valid Prediction Time (VPT) Experiment
Evaluates the "Lead Time" or VPT for predicting system disruptions in the Lorenz '63 system.
```bash
python VPT_Experiment/vpt_experiment.py
```
**Output:** `VPT_Experiment/VPT_Comparison_QRC_CRC.png`

