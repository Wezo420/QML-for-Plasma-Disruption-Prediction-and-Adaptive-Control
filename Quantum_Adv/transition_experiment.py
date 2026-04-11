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

# 1. Generate Transition Dataset
dt = 0.01
train_steps = 3000
test_steps = 3000
q0 = np.array([7.432487609628195, 10.02071718705213, 29.62297428638419])
N_transient = 1000

system = Systems(dt, train_steps, q0, 1, N_transient)
system.set_param_lorenz63()

def gen_drifting_rho_data(steps, rho_start, rho_end, initial_q):
    rho_values = np.linspace(rho_start, rho_end, steps)
    data = []
    curr_q = initial_q.copy()
    
    # RK4 implementation that allows changing r at each step
    sigma = 10.0
    b = 8/3
    
    for r in rho_values:
        def lorenz_deriv(u, rho_val):
            x, y, z = u
            return np.array([sigma * (y - x), x * (rho_val - z) - y, x * y - b * z])
        
        k1 = dt * lorenz_deriv(curr_q, r)
        k2 = dt * lorenz_deriv(curr_q + k1/2, r)
        k3 = dt * lorenz_deriv(curr_q + k2/2, r)
        k4 = dt * lorenz_deriv(curr_q + k3, r)
        curr_q = curr_q + (k1 + 2*k2 + 2*k3 + k4)/6
        data.append(curr_q.copy())
    return np.array(data), rho_values

print("Generating Training Data...")
# Warm up at rho=15
warmup_data, _ = gen_drifting_rho_data(N_transient, 15, 15, q0)
# Transition data for training
U_train_raw, rho_train = gen_drifting_rho_data(train_steps, 15, 30, warmup_data[-1])

# Target Y: 1 / (1 + distance to 24.74) for a "spike" at transition
critical_rho = 24.74
# We use a target that spikes at the critical point
Y_train_signal = (1.0 / (1.0 + np.abs(rho_train - critical_rho))).reshape(-1, 1)

# Scaling Lorenz data
scaler = MinMaxScaler((0, 1))
U_train = scaler.fit_transform(U_train_raw)

# 2. Setup QRC
qubits = 5 # Reduced for speed in simulation
N_units = 2**qubits
dim = 3
rho_q = 0.9
epsilon_q = 0.5
sigma_in_q = 0.1
tikh_q = np.array([1e-6])
bias_in = np.array([0.0])
bias_out = np.array([1.0])
config = 1
emulator = "sv_sim"
shots = 1024
snapshots = 1

qrc = QuantumReservoirNetwork(rho_q, epsilon_q, sigma_in_q, tikh_q, bias_in, bias_out, qubits, N_units, dim, config, emulator, shots, snapshots)
qrc.method_qc(parameterized=False)
alpha = qrc.gen_random_unitary(seed=42, range=np.pi)

# Pad Y to dim=3 to match qrc requirements
Y_train_padded = np.zeros((train_steps, 3))
Y_train_padded[:, 0] = Y_train_signal[:, 0]

# Reshape for quantum_training (expects UU: (N_ts, N_steps, dim))
U_train_q = U_train.reshape(1, train_steps, 3)
Y_train_q = Y_train_padded.reshape(1, train_steps, 3)

# Washout
N_washout = 500
U_washout = U_train_q[:, :N_washout, :]
U_train_actual = U_train_q[:, N_washout:, :]
Y_train_actual = Y_train_q[:, N_washout:, :]

print("Training QRC (this may take a minute)...")
Xa, Wout_q, LHS, RHS = qrc.quantum_training(U_washout, U_train_actual, Y_train_actual, alpha)

# 2.5 Setup CRC
print("Training CRC...")
density = 0.5
sigma_in = 0.1
rho = 0.9
epsilon = 0.5
tikh = np.array([1e-6])

crc = EchoStateNetwork(tikh, sigma_in, rho, epsilon, bias_in, bias_out, N_units, dim, density)
crc.norm_u = 1.0 # Since data is already scaled
Win = crc.gen_input_matrix(seed=42)
W = crc.gen_reservoir_matrix(seed=42)

Xa_crc, Wout_crc, LHS_crc, RHS_crc = crc.train(U_washout, U_train_actual, Y_train_actual, Win, W)


# 3. Test on drifting rho
print("Generating Test Data (simulating plasma pulse)...")
test_warmup, _ = gen_drifting_rho_data(N_transient, 15, 15, q0)
U_test_raw, rho_test = gen_drifting_rho_data(test_steps, 15, 30, test_warmup[-1])
U_test = scaler.transform(U_test_raw)

print("Predicting Warning Signal with QRC...")
xf_washout = qrc.quantum_openloop(U_test[:N_washout], np.zeros(N_units), alpha)[-1, :N_units]
Xa_test = qrc.quantum_openloop(U_test[N_washout:], xf_washout, alpha)
warning_signal = np.dot(Xa_test[1:], Wout_q[0])[:, 0]

print("Predicting Warning Signal with CRC...")
xf_washout_crc = crc.open_loop(U_test[:N_washout], np.zeros(N_units), Win, W)[-1, :N_units]
Xa_test_crc = crc.open_loop(U_test[N_washout:], xf_washout_crc, Win, W)
warning_signal_crc = np.dot(Xa_test_crc[1:], Wout_crc[0])[:, 0]


# 4. Plots
plt.figure(figsize=(12, 12))

# Plot Lorenz variable X
plt.subplot(4, 1, 1)
plt.plot(U_test_raw[:, 0], label='Lorenz x-variable', color='blue')
critical_idx = np.where(rho_test >= critical_rho)[0][0]
plt.axvline(x=critical_idx, color='red', linestyle='--', label='Tipping Point (rho=24.74)')
plt.title("Lorenz Dynamics during Parameter Drift")
plt.ylabel("Value")
plt.legend()

# Plot Rho parameter
plt.subplot(4, 1, 2)
plt.plot(rho_test, label='Rho parameter', color='green')
plt.axhline(y=critical_rho, color='red', linestyle='--', label='Critical Threshold')
plt.ylabel("Rho")
plt.legend()

# Plot Warning Signal
plt.subplot(4, 1, 3)
time_axis = np.arange(N_washout, test_steps)
plt.plot(time_axis, warning_signal, label='QRC Warning Signal', color='orange', linewidth=2)
plt.plot(time_axis, warning_signal_crc, label='CRC Warning Signal', color='purple', linewidth=2, linestyle=':')
plt.axvline(x=critical_idx, color='red', linestyle='--', label='Actual Tipping Point')
plt.title("Early Warning Signal Prediction (QRC vs CRC)")
plt.ylabel("Signal Intensity")
plt.legend()

# Plot zoom around transition
plt.subplot(4, 1, 4)
zoom_start = max(0, critical_idx - 500)
zoom_end = min(test_steps, critical_idx + 500)
plt.plot(time_axis, warning_signal, label='QRC', color='orange', linewidth=2)
plt.plot(time_axis, warning_signal_crc, label='CRC', color='purple', linewidth=2, linestyle=':')
plt.axvline(x=critical_idx, color='red', linestyle='--', label='Tipping Point')
plt.xlim(zoom_start, zoom_end)
plt.title("Zoomed Warning Signal (Rising before Tipping Point)")
plt.xlabel("Time Steps")
plt.ylabel("Signal Intensity")
plt.legend()

plt.tight_layout()
plt.savefig("transition_results_comparison.png")
print("Experiment complete. Results saved to transition_results_comparison.png")
