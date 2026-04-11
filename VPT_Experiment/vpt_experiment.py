import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the QRC directory to path
sys.path.append(os.path.join(os.getcwd(), 'Stability_QRC_GS-main', 'Stability_QRC_GS-main', 'src'))

from QRC.systems import Systems
from QRC.qrc import QuantumReservoirNetwork
from QRC.crc import EchoStateNetwork
from sklearn.preprocessing import MinMaxScaler

# --- 1. Constants and VPT Evaluation Function ---
# 1. Define the maximum acceptable error threshold
# Usually 0.4 or 15-20% of the attractor's standard deviation
ERROR_THRESHOLD = 0.4 

# 2. Define the Lyapunov Time for Lorenz '63
# The maximum Lyapunov exponent for standard Lorenz (rho=28) is ~0.905.
# Lyapunov Time = 1 / lambda_max
lambda_max = 0.905 
lyapunov_time = 1.0 / lambda_max

dt = 0.01 

def calculate_vpt(y_true, y_pred, dt, lyapunov_time, threshold=ERROR_THRESHOLD):
    """
    Calculates the Valid Prediction Time (VPT) in units of Lyapunov Time.
    y_true: Ground truth trajectory array of shape (time_steps, features)
    y_pred: Model predicted trajectory array of shape (time_steps, features)
    """
    # Calculate Euclidean distance (error) at each time step
    error = np.linalg.norm(y_true - y_pred, axis=1)
    
    # Find the first time step where the error exceeds the threshold
    exceed_indices = np.where(error > threshold)[0]
    
    if len(exceed_indices) == 0:
        # If it never exceeds the threshold, it predicted the whole window perfectly
        valid_steps = y_true.shape[0]
    else:
        # The first index where it broke the threshold
        valid_steps = exceed_indices[0]
        
    # Convert steps to physical time, then to Lyapunov times
    physical_time = valid_steps * dt
    vpt_lyapunov = physical_time / lyapunov_time
    
    return vpt_lyapunov, error


# --- 2. Data Generation ---
tot_steps = 4100
q0 = np.array([7.432487609628195, 10.02071718705213, 29.62297428638419])
N_transient = 1000

system = Systems(dt, tot_steps, q0, 1, N_transient)
system.set_param_lorenz63() 
print("Generating clean Lorenz data...")
UU_raw = system.gen_data_lorenz63()[0] 

scaler = MinMaxScaler((0, 1))
UU_clean = scaler.fit_transform(UU_raw)

N_ts = 1
N_washout = 500
N_train = 2000
prediction_steps = 500

UU_clean_q = UU_clean.reshape(1, UU_clean.shape[0], 3)

U_washout = UU_clean_q[:, :N_washout, :]
U_train = UU_clean_q[:, N_washout:N_washout+N_train, :]
Y_train = UU_clean_q[:, N_washout+1:N_washout+N_train+1, :]

# --- 3. Train Models ---
# Common Parameters
dim = 3
bias_in = np.array([0.0])
bias_out = np.array([1.0])

# QRC Setup
qubits = 5
N_units = 2**qubits
rho_q = 0.9
epsilon_q = 0.5
sigma_in_q = 0.1
tikh_array = np.array([1e-6])

print("Training QRC...")
qrc = QuantumReservoirNetwork(rho_q, epsilon_q, sigma_in_q, tikh_array, bias_in, bias_out, qubits, N_units, dim, 1, "sv_sim", 1024, 1)
qrc.method_qc(parameterized=False)
alpha = qrc.gen_random_unitary(seed=42, range=np.pi)
Xa_qrc, Wout_qrc, _, _ = qrc.quantum_training(U_washout, U_train, Y_train, alpha)

# CRC Setup
print("Training CRC...")
crc = EchoStateNetwork(tikh_array, sigma_in_q, rho_q, epsilon_q, bias_in, bias_out, N_units, dim, 0.5)
crc.norm_u = 1.0
Win = crc.gen_input_matrix(seed=42)
W = crc.gen_reservoir_matrix(seed=42)
Xa_crc, Wout_crc, _, _ = crc.train(U_washout, U_train, Y_train, Win, W)


# --- 4. Autonomous Generative Prediction (Closed-Loop) ---
print("Running Autonomous Predictions...")

# Get true continuation for comparison
# Note: shape is (prediction_steps, 3)
Y_true_future = UU_clean_q[0, N_washout+N_train+1 : N_washout+N_train+1+prediction_steps, :]

# QRC Autonomous
x0_qrc = Xa_qrc[0, -1] # Last augmented reservoir state from training
# quantum_closedloop returns Yh (time series of prediction) and Xa (final state)
# Yh shape is (N+1, dim) where Yh[0] is the prediction for the first step.
Y_pred_qrc_all, _ = qrc.quantum_closedloop(prediction_steps, x0_qrc, Wout_qrc[0], alpha)
Y_pred_qrc = Y_pred_qrc_all[:-1] # Remove the extra last step to match prediction_steps length

# CRC Autonomous
x0_crc = Xa_crc[0, -1] # Last augmented reservoir state from training
Y_pred_crc_all, _ = crc.closed_loop(prediction_steps, x0_crc, Wout_crc[0], Win, W)
Y_pred_crc = Y_pred_crc_all[:-1] 

# Calculate VPT
vpt_qrc, error_qrc = calculate_vpt(Y_true_future, Y_pred_qrc, dt, lyapunov_time)
vpt_crc, error_crc = calculate_vpt(Y_true_future, Y_pred_crc, dt, lyapunov_time)

print(f"Classical Reservoir VPT: {vpt_crc:.2f} Lyapunov Times")
print(f"Quantum Reservoir VPT: {vpt_qrc:.2f} Lyapunov Times")


# --- 5. Visualization ---
time_axis = np.arange(prediction_steps) * dt / lyapunov_time

plt.figure(figsize=(12, 6))

# Plot the errors over time
plt.plot(time_axis, error_crc, 'r-', label=f'CRC Error (VPT = {vpt_crc:.2f})', linewidth=1.5)
plt.plot(time_axis, error_qrc, 'b-', label=f'QRC Error (VPT = {vpt_qrc:.2f})', linewidth=1.5)

# Plot the failure threshold
plt.axhline(y=ERROR_THRESHOLD, color='k', linestyle='--', label='Failure Threshold (Disruption Occurs)')

# Highlight the VPT points
plt.axvline(x=vpt_crc, color='r', linestyle=':', alpha=0.7)
plt.axvline(x=vpt_qrc, color='b', linestyle=':', alpha=0.7)

plt.title('Valid Prediction Time (Lead Time) for Disruption Mitigation', fontsize=16)
plt.xlabel('Prediction Horizon (Lyapunov Times)', fontsize=14)
plt.ylabel('Trajectory Divergence (Euclidean Error)', fontsize=14)
plt.yscale('log')

# Zoom in on the relevant area
max_vpt = max(vpt_crc, vpt_qrc)
if max_vpt > 0:
    plt.xlim(0, max_vpt + 1)
else:
    plt.xlim(0, 5)

plt.legend(fontsize=12, loc='upper left')
plt.grid(True, which="both", ls="--", alpha=0.5)

plot_path = os.path.join('VPT_Experiment', 'VPT_Comparison_QRC_CRC.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nExperiment complete. Plot saved to {plot_path}")
plt.show()