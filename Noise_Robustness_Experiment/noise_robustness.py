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

# 1. Noise Function Definition
def add_gaussian_noise(data, noise_percentage):
    """
    Adds Gaussian noise to the dataset.
    noise_percentage: e.g., 0.05 for 5% noise.
    """
    if noise_percentage == 0.0:
        return data.copy()
        
    # Calculate the standard deviation of the original data
    std_dev = np.std(data, axis=1, keepdims=True) if data.ndim > 1 else np.std(data)
    
    # Generate noise with mean 0 and standard deviation scaled by the percentage
    noise = np.random.normal(0, std_dev * noise_percentage, data.shape)
    
    return data + noise

# 2. Data Generation (Standard Lorenz 63)
dt = 0.01
tot_steps = 4100
q0 = np.array([7.432487609628195, 10.02071718705213, 29.62297428638419])
N_transient = 1000

system = Systems(dt, tot_steps, q0, 1, N_transient)
system.set_param_lorenz63() # sigma=10, r=28, b=8/3
print("Generating clean Lorenz data...")
UU_raw = system.gen_data_lorenz63()[0] # (tot_steps - N_transient, 3)

# Scaling clean data
scaler = MinMaxScaler((0, 1))
UU_clean = scaler.fit_transform(UU_raw)

# Training Parameters
N_ts = 1
N_washout = 500
N_train = 2000
N_test = 500

# Reshape for reservoir functions (expects (N_ts, steps, dim))
UU_clean_q = UU_clean.reshape(1, UU_clean.shape[0], 3)

# 3. Model Parameters
qubits = 5
N_units = 2**qubits
dim = 3
rho_q = 0.9
epsilon_q = 0.5
sigma_in_q = 0.1
tikh_val = 1e-6
tikh_array = np.array([tikh_val])
bias_in = np.array([0.0])
bias_out = np.array([1.0])

# 4. Noise Sweep Loop
noise_levels = [0.0, 0.01, 0.03, 0.05, 0.10]
mse_crc_results = []
mse_qrc_results = []

for noise in noise_levels:
    print(f"\n--- Testing Noise Level: {noise * 100}% ---")
    
    # 4.1 Corrupt the input data
    noisy_UU = add_gaussian_noise(UU_clean_q, noise)
    
    # 4.2 Split into Washout and Train sets
    # U_in (noisy) must predict clean next state
    U_washout_noisy = noisy_UU[:, :N_washout, :]
    U_train_noisy = noisy_UU[:, N_washout:N_washout+N_train, :]
    Y_train_clean = UU_clean_q[:, N_washout+1:N_washout+N_train+1, :]
    
    # Test set
    U_test_noisy = noisy_UU[0, N_washout+N_train : N_washout+N_train+N_test]
    # Y_test_actual is the clean state corresponding to the NEXT step of each U_test_noisy
    Y_test_actual = UU_clean_q[0, N_washout+N_train+1 : N_washout+N_train+N_test+1]

    # --- RUN QUANTUM RESERVOIR (QRC) ---
    print("Training QRC...")
    qrc = QuantumReservoirNetwork(rho_q, epsilon_q, sigma_in_q, tikh_array, bias_in, bias_out, qubits, N_units, dim, 1, "sv_sim", 1024, 1)
    qrc.method_qc(parameterized=False)
    alpha = qrc.gen_random_unitary(seed=42, range=np.pi)
    
    Xa_qrc, Wout_qrc, _, _ = qrc.quantum_training(U_washout_noisy, U_train_noisy, Y_train_clean, alpha)
    
    print("Predicting QRC...")
    xf_last_train_qrc = Xa_qrc[0, -1, :N_units]
    Xa_test_qrc = qrc.quantum_openloop(U_test_noisy, xf_last_train_qrc, alpha)
    # Xa_test_qrc has N_test+1 states. Index 0 is the state BEFORE first test input.
    # States 1 to N_test are the states AFTER each test input.
    qrc_predictions = np.dot(Xa_test_qrc[1:], Wout_qrc[0])
    
    mse_qrc = np.mean((qrc_predictions - Y_test_actual)**2)
    mse_qrc_results.append(mse_qrc)
    print(f"QRC MSE: {mse_qrc:.2e}")

    # --- RUN CLASSICAL RESERVOIR (CRC) ---
    print("Training CRC...")
    crc = EchoStateNetwork(tikh_array, sigma_in_q, rho_q, epsilon_q, bias_in, bias_out, N_units, dim, 0.5)
    crc.norm_u = 1.0
    Win = crc.gen_input_matrix(seed=42)
    W = crc.gen_reservoir_matrix(seed=42)
    
    Xa_crc, Wout_crc, _, _ = crc.train(U_washout_noisy, U_train_noisy, Y_train_clean, Win, W)
    
    print("Predicting CRC...")
    xf_last_train_crc = Xa_crc[0, -1, :N_units]
    Xa_test_crc = crc.open_loop(U_test_noisy, xf_last_train_crc, Win, W)
    crc_predictions = np.dot(Xa_test_crc[1:], Wout_crc[0])
    
    mse_crc = np.mean((crc_predictions - Y_test_actual)**2)
    mse_crc_results.append(mse_crc)
    print(f"CRC MSE: {mse_crc:.2e}")

# 5. Visualize the "Quantum Advantage"
plt.figure(figsize=(10, 6))
plt.plot([n*100 for n in noise_levels], mse_crc_results, 'r-o', label='Classical Reservoir (ESN)', linewidth=2)
plt.plot([n*100 for n in noise_levels], mse_qrc_results, 'b-s', label='Quantum Reservoir (QRC)', linewidth=2)

plt.title('Robustness Against Diagnostic Noise (Lorenz 63 Proxy)', fontsize=16)
plt.xlabel('Gaussian Noise Level (%)', fontsize=14)
plt.ylabel('Mean Squared Error (Prediction Accuracy)', fontsize=14)
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.legend(fontsize=12)

plot_path = os.path.join('Noise_Robustness_Experiment', 'Noise_Robustness_QRC_vs_CRC.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nExperiment complete. Plot saved to {plot_path}")
plt.show()
