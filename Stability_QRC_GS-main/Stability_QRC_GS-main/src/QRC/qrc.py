
import numpy as np
from qiskit import  QuantumCircuit, QuantumRegister, ClassicalRegister , transpile , qpy
from qiskit_aer import Aer
from scipy.signal import savgol_filter
from QRC.unitaryblock import Unitary4 , Unitary_FullyEnt , Unitary_FullyEntSym , Unitary_Feature , Unitary_Linear , Unitary_Linear_New
from qiskit.circuit import ParameterVector
from qiskit_ibm_runtime.fake_provider import FakeBrisbane , FakeKyoto
from qiskit_aer import AerSimulator
import warnings
import scipy

class QuantumReservoirNetwork:
    def __init__(self,rho_q,epsilon_q,sigma_in_q,tikh_q,bias_in,bias_out,qubits,N_units,dim,config,emulator,shots,snapshots):
        self.rho_q      = rho_q
        self.epsilon_q  = epsilon_q
        self.sigma_in_q = sigma_in_q
        self.tikh_q     = tikh_q
        self.bias_in    = bias_in
        self.bias_out   = bias_out
        self.qubits     = qubits
        self.N_units    = N_units
        self.dim        = dim
        self.config     = config
        self.emulator   = emulator
        self.shots      = shots
        self.snapshots  = snapshots
        self.alpha      = None
        self.param_qc   = None
        self.transpiled_qc = None
        self.transpiled_warn = None


    @property
    def rho_q(self):
        return (self._rho_q)

    @rho_q.setter
    def rho_q (self,value):
        self._rho_q = value

    @property
    def tikh_q(self):
        return (self._tikh_q)

    @tikh_q.setter
    def tikh_q (self,value):
        self._tikh_q = value

    @property
    def epsilon_q(self):
        return (self._epsilon_q)

    @epsilon_q.setter
    def epsilon_q (self,value):
        self._epsilon_q = value

    @property
    def qubits(self):
        return (self._qubits)

    @qubits.setter
    def qubits (self,value):
        self._qubits = value

    @property
    def config(self):
        return (self._config)

    @config.setter
    def config (self,value):
        self._config = value

    @property
    def shots(self):
        return (self._shots)

    @shots.setter
    def shots (self,value):
        self._shots = value

    @property
    def emulator(self):
        return (self._emulator)

    @emulator.setter
    def emulator (self,value):
        self._emulator = value

    @property
    def snapshots(self):
        return (self._snapshots)

    @snapshots.setter
    def snapshots (self,value):
        self._snapshots = value

    def gen_random_unitary(self,seed,range):
        rnd = np.random.RandomState(seed)
        alpha = np.zeros(self.qubits)
        alpha_ins = rnd.uniform(0,range,size = (self.qubits)) # Beta as uniform distribution
        alpha     = alpha_ins

        return alpha

    def gen_param_quantumcircuit(self):

        P  = ParameterVector('P'    , self.N_units)
        X  = ParameterVector('X'    , self.dim)
        A  = ParameterVector('alpha', self.qubits)

        q_r = QuantumRegister(self.qubits)
        c   = ClassicalRegister(self.qubits)
        qc  = QuantumCircuit(q_r,c) # Quantum Register , Classical Bits


        if self.config == 1:
            print('Configuration 1')
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary4(self.qubits,P,qc,'P')
            qc.barrier()

            Unitary_FullyEnt(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_FullyEntSym(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 2:
            print('Configuration 2')
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary4(self.qubits,P,qc,'P')
            qc.barrier()

            Unitary4(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary4(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 3:
            print('Configuration 3')
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary4(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary4(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary4(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 4:
            print('Configuration 4')
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary_FullyEnt(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_FullyEnt(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_FullyEntSym(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 5:
            print('Configuration 5')
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary_Feature(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_Feature(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary4(self.qubits,A,qc,'A')
            qc.barrier()

        self.param_qc = qc

        return self.param_qc

    def gen_quantumcircuit(self,prob_p,x_in,alpha):

        P  = prob_p
        X  = x_in
        A  = alpha

        q_r = QuantumRegister(self.qubits)
        c   = ClassicalRegister(self.qubits)
        qc  = QuantumCircuit(q_r,c) # Quantum Register , Classical Bits

        if self.config == 1:
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary4(self.qubits,P,qc,'P')
            qc.barrier()

            Unitary_FullyEnt(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_FullyEntSym(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 2:
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary4(self.qubits,P,qc,'P')
            qc.barrier()

            Unitary4(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary4(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 3:
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary4(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary4(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary4(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 4:

            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary_FullyEnt(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_FullyEnt(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_FullyEntSym(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 5:
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary_Feature(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_Feature(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary4(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 6:
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary_Feature(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_Feature(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_FullyEntSym(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 7:

            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary_Linear(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_Linear(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_FullyEntSym(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 8:
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary_Linear_New(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_Linear_New(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_FullyEntSym(self.qubits,A,qc,'A')
            qc.barrier()

        if self.config == 9:
            hadamard = min(self.dim,self.qubits)
            qc.h(range(hadamard))

            Unitary_Linear_New(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_Linear_New(self.qubits,X,qc,'X')
            qc.barrier()

            Unitary_Linear_New(self.qubits,A,qc,'A')
            qc.barrier()

        self.qc = qc

        return self.qc


    def method_qc(self,parameterized=False):
        self.parameterized = parameterized
        return self.parameterized

    def quantum_step(self,prob_p,x_in,alpha):

        if self.parameterized:
            self.qc = self.load_quantumcircuit(prob_p,x_in,alpha)
        else:
            self.qc = self.gen_quantumcircuit(prob_p,x_in,alpha)

        if self.emulator  == "sv_sim":
            simulator = Aer.get_backend('aer_simulator_statevector')
            #simulator.set_options(device='GPU')
            self.qc.save_statevector()
            #qc = transpile(qc, simulator)
            result      = simulator.run(self.qc).result()
            statevector = result.get_statevector(self.qc)
            prob_tilde  = np.abs(np.array(statevector)**2)

        elif self.emulator == "fake_kyoto":
            self.qc.measure_active()

            # Fake Brisbane
            backend = FakeKyoto()
            simulator = AerSimulator.from_backend(backend)

            # Else same as qasm_simulator
            simulator = simulator.run(self.qc,shots=self.shots)
            result    = simulator.result()
            counts    = result.get_counts(self.qc)

            a=list(np.zeros(self.N_units))

            for ll in range(self.N_units):
                a[ll]=f'{ll:0{self.qubits}b}'

            # Turning count in terms of probabilities
            psi_tilde = {}
            for output in list(a):
                if output in counts:
                    psi_tilde[output] = counts[output]/self.shots
                else:
                    psi_tilde[output] = 0

            #psi_tilde = dict(sorted(psi_tilde.items())) # sorting dictionary with binary number 0s ---> higher

            psi_tilde = np.array([j for j in psi_tilde.values()]) #Takes values of probabilities from the dictionary

            prob_tilde = (psi_tilde)

        elif self.emulator == "qasm_sim":

            self.qc.measure_all(add_bits=False)
            simulator = Aer.get_backend('qasm_simulator')
            simulator = simulator.run(self.qc,shots=self.shots)
            result    = simulator.result()
            counts    = result.get_counts(self.qc)

            a=list(np.zeros(self.N_units))

            for ll in range(self.N_units):
                a[ll]=f'{ll:0{self.qubits}b}'

            # Turning count in terms of probabilities
            psi_tilde = {}
            for output in list(a):
                if output in counts:
                    psi_tilde[output] = counts[output]/self.shots
                else:
                    psi_tilde[output] = 0

            #psi_tilde = dict(sorted(psi_tilde.items())) # sorting dictionary with binary number 0s ---> higher

            psi_tilde = np.array([j for j in psi_tilde.values()]) #Takes values of probabilities from the dictionary

            prob_tilde = (psi_tilde)


        else:
            raise AttributeError("Please select a valid emulator from the list, (a) sv_sim (b) qasm_sim (c) fake_kyoto")


        prob_neweps = (1-self.epsilon_q)*prob_p+self.epsilon_q*prob_tilde # including epsilon_q/leaking rate

        # x_in_new = np.dot(Wout_q.T,prob_p) # Prediction Step
        # For next step
        prob_neweps = np.hstack((prob_neweps, self.bias_out)) ###Bias out

        return prob_neweps


    def quantum_openloop(self,U, x0,alpha):
        """ Advances QESN in open-loop.
            Args:
                U: input time series
                x0: initial reservoir state
            Returns:
                time series of augmented reservoir states
        """
        N     = U.shape[0]
        Xa    = np.empty((N+1, self.N_units+1))
        Xa[0] = np.concatenate((x0,self.bias_out))

        for i in np.arange(1,N+1):
            Xa[i] = self.quantum_step(Xa[i-1,:self.N_units], U[i-1],alpha)

        return Xa
    

    def quantum_closedloop(self,N,x0,Wout_q,alpha):
        """Advances Quantum Circ in Closed Loop
        Args:
            N : Number of Time Steps
            x0 : Initial Reservoir State
            Wout_q : Output Matrix

        Returns:
            Yh: Time Series of Prediction
            Xa: Final Augmented Reservoir State
        """
        xa    = x0.copy()
        Yh    = np.empty((N+1, self.dim))
        Yh[0] = np.dot(xa, Wout_q)
        for i in np.arange(1,N+1):
            xa    =  self.quantum_step(xa[:self.N_units], Yh[i-1],alpha)
            Yh[i] =  np.dot(xa, Wout_q) #np.linalg.multi_dot([xa, Wout_q])

        return Yh, xa


    def quantum_training(self,U_washout, U_train, Y_train,alpha):
        """ Trains QESN.
            Args:
                U_washout: washout input time series
                U_train: training input time series
                tikh_q: tikhonov factor
            Returns:
                time series of augmented reservoir states
                optimal output matrix
        """

        LHS = 0
        RHS = 0

        N  = U_train[0].shape[0]
        Xa  = np.zeros((U_washout.shape[0], N+1, self.N_units+1))

        for i in range(U_washout.shape[0]): # if training on multiple time-series

            ## washout phase
            xf_washout =  self.quantum_openloop(U_washout[i], np.zeros(self.N_units),alpha)[-1,:self.N_units]
            #xf_washout = quant_open_loop(U_washout[i], np.zeros(N_units_q), sigma_in_q, rho_q,beta,epsilon_q)[-1,:N_units]

            ## open-loop train phase
            Xa[i] = self.quantum_openloop(U_train[i], xf_washout,alpha)
            #Xa[i] = quant_open_loop(U_train[i], xf_washout, sigma_in_q, rho_q,beta,epsilon_q)

            ## Ridge Regression
            LHS  += np.dot(Xa[i,1:].T, Xa[i,1:])
            RHS  += np.dot(Xa[i,1:].T, Y_train[i])

        #solve linear system for each tikh_qonov parameter
        Wout_q = np.zeros((self.tikh_q.size, self.N_units+1,self.dim))
        for j in np.arange(self.tikh_q.size):
            Wout_q[j] = np.linalg.solve(LHS + self.tikh_q[j]*np.eye(self.N_units+1), RHS)
        return Xa, Wout_q, LHS, RHS

    def quantum_openloop_denoised(self,U, x0,alpha,freq,poly):
        """ Advances QESN in open-loop.
            Args:
                U: input time series
                x0: initial reservoir state
            Returns:
                time series of augmented reservoir states
        """
        N     = U.shape[0]
        Xa    = np.empty((N+1, self.N_units+1))
        Xa[0] = np.concatenate((x0,self.bias_out))


        Xa_dn    = np.empty((N+1, self.N_units+1))
        x0_dn    = savgol_filter(x0, window_length = freq, polyorder=poly)
        Xa_dn[0] = np.concatenate((x0_dn,self.bias_out))

        for i in np.arange(1,N+1):
            Xa[i] = self.quantum_step(Xa[i-1,:self.N_units], U[i-1],alpha)

        Xa_dn = Xa_dn.T # reshaping for singal denoise
        for m in range(1,self.N_units+1):
            Xa_dn[m] = savgol_filter(Xa.T[m], window_length = freq, polyorder=poly) # removing output bias before filter
            # Xa_dn[i] = np.concatenate((Xa_dn[m][:-1],self.bias_out))

        Xa_dn = Xa_dn.T # reshaping back for training

        return Xa , Xa_dn

    def quantum_training_denoised(self,U_washout, U_train, Y_train,alpha,freq,poly):
        """ Trains QESN.
            Args:
                U_washout: washout input time series
                U_train: training input time series
                tikh_q: tikh_qonov factor
            Returns:
                time series of augmented reservoir states
                optimal output matrix
        """

        LHS = 0
        RHS = 0

        LHS_dn = 0
        RHS_dn = 0

        N  = U_train[0].shape[0]
        Xa  = np.zeros((U_washout.shape[0], N+1, self.N_units+1))
        Xa_dn  = np.zeros((U_washout.shape[0], N+1, self.N_units+1))

        for i in range(U_washout.shape[0]):

            ## washout phase
            xf_washout , xf_washout_dn =  self.quantum_openloop_denoised(U_washout[i], np.zeros(self.N_units),alpha,freq,poly)

            xf_washout = xf_washout[-1,:self.N_units]
            xf_washout_dn = xf_washout_dn[-1,:self.N_units]


            ## open-loop train phase
            Xa[i] , Xa_dn[i] = self.quantum_openloop_denoised(U_train[i], xf_washout,alpha,freq,poly)
            #Xa[i] = quant_open_loop(U_train[i], xf_washout, sigma_in_q, rho_q,beta,epsilon_q)

            ## Ridge Regression
            LHS  += np.dot(Xa[i,1:].T, Xa[i,1:])
            RHS  += np.dot(Xa[i,1:].T, Y_train[i])

            LHS_dn  += np.dot(Xa_dn[i,1:].T, Xa_dn[i,1:])
            RHS_dn  += np.dot(Xa_dn[i,1:].T, Y_train[i])

        #solve linear system for each tikh_qonov parameter
        Wout_q = np.zeros((self.tikh_q.size, self.N_units+1,self.dim))
        Wout_q_dn = np.zeros((self.tikh_q.size, self.N_units+1,self.dim))


        for j in np.arange(self.tikh_q.size):
            Wout_q[j] = np.linalg.solve(LHS + self.tikh_q[j]*np.eye(self.N_units+1), RHS)
            Wout_q_dn[j] = np.linalg.solve(LHS_dn + self.tikh_q[j]*np.eye(self.N_units+1), RHS_dn)

        return Xa, Wout_q, LHS, RHS , Xa_dn , Wout_q_dn , LHS_dn, RHS_dn



    ### Lyapunov Spectrum Calculation


    # def const_jacobian(self,Win,W,Wout,norm):
    #     dfdu = np.r_[np.diag(self.sigma_in_q/norm),[np.zeros(self.dim)]]

    #     #dfdu = self.epsilon * np.multiply(Win[:, : self.dim], 1.0 / norm[: self.dim])

    #     #                 self._dfdu_const = self.alpha * self.W_in[:, : self.N_dim].multiply(
    #     #             1.0 / self.norm_in[1][: self.N_dim]
    #     #print(dfdu.shape)
    #     d    = Win.dot(dfdu)
    #     c    = np.matmul(d,Wout[:self.N_units,:].T)

    #     return c, W.dot(np.diag(np.ones(self.N_units)*self.rho_q))

    def quantum_step_rfqrc(self,prob_p,x_in1,x_in2,alpha,layered):
        # x_const = same input but unshifted

        P  = prob_p
        X1  = x_in1 # one to constant, one to vary
        X2  = x_in2
        A  = alpha

        q_r = QuantumRegister(self.qubits)
        c   = ClassicalRegister(self.qubits)
        qc  = QuantumCircuit(q_r,c) # Quantum Register , Classical Bits

        hadamard = min(self.dim,self.qubits)
        qc.h(range(hadamard))

        Unitary_FullyEnt(self.qubits,X1,qc,'X1')
        qc.barrier()

        if layered:
            Unitary_FullyEnt(self.qubits,X2,qc,'X2')
            qc.barrier()
            #print('Layer added')

        Unitary_FullyEntSym(self.qubits,A,qc,'A')
        qc.barrier()

        self.qc = qc

        if self.emulator  == "sv_sim":
            simulator = Aer.get_backend('aer_simulator_statevector')
            #simulator.set_options(device='GPU')
            self.qc.save_statevector()
            #qc = transpile(qc, simulator)
            result      = simulator.run(self.qc).result()
            statevector = result.get_statevector(self.qc)
            prob_tilde  = np.abs(np.array(statevector)**2)

        elif self.emulator == "qasm_sim":

            self.qc.measure_all(add_bits=False)
            simulator = Aer.get_backend('qasm_simulator')
            simulator = simulator.run(self.qc,shots=self.shots)
            result    = simulator.result()
            counts    = result.get_counts(self.qc)

            a=list(np.zeros(self.N_units))

            for ll in range(self.N_units):
                a[ll]=f'{ll:0{self.qubits}b}'

            # Turning count in terms of probabilities
            psi_tilde = {}
            for output in list(a):
                if output in counts:
                    psi_tilde[output] = counts[output]/self.shots
                else:
                    psi_tilde[output] = 0

            #psi_tilde = dict(sorted(psi_tilde.items())) # sorting dictionary with binary number 0s ---> higher

            psi_tilde = np.array([j for j in psi_tilde.values()]) #Takes values of probabilities from the dictionary

            prob_tilde = (psi_tilde)


        else:
            raise AttributeError("Please select a valid emulator from the list, (a) sv_sim (b) qasm_sim (c) fake_kyoto")


        #prob_neweps = (1-self.epsilon_q)*prob_p+self.epsilon_q*prob_tilde # including epsilon_q/leaking rate

        prob_neweps = prob_tilde
        # x_in_new = np.dot(Wout_q.T,prob_p) # Prediction Step
        # For next step
        prob_neweps = np.hstack((prob_neweps, self.bias_out)) ###Bias out

        return prob_neweps


    def leak_integral(self, x, x_prev):
        """Derivative of the tanh part
        This derivative appears in different gradient calculations
        So, it makes sense to calculate it only once and call as required

        Args:
        x: reservoir states at time i+1, x(i+1)
        x_prev: reservoir states at time i, x(i)
        """
        # first we find tanh(...)
        #print(x.shape,x_prev.shape)
        x_tilde = (x - (1 - self.epsilon_q) * x_prev) / self.epsilon_q
        #print(x_tilde.shape)
        # derivative of tanh(...) is 1-tanh**2(...)
        #dtanh = 1.0 - x_tilde**2

        return x_tilde

    def quantum_gradient_calc_wrty(self,xa,xa_prev,Yh,alpha,layered):
        """_summary_

        Args:
            xa (_type_): _description_
            Yh (_type_): _description_
            alpha (_type_): _description_

        Returns:
            _xa_grad_: _Gradient wrt input parameter_
        """
        Yh_orig = Yh.copy()
        x_tilde = self.leak_integral(xa,xa_prev)
        # x_tilde = xa.copy()
        xa_y_grad = np.zeros([self.N_units,self.dim])

        for ll in range(self.dim):
            # Quantum gradient wrt x
            Yh_plus = Yh.copy()
            Yh_plus[ll] = Yh[ll]+(np.pi/2)

            Yh_minus= Yh.copy()
            Yh_minus[ll] = Yh[ll]-(np.pi/2)

            if layered:
                xa_y_plus_1    = self.quantum_step_rfqrc(x_tilde, Yh_plus, Yh_orig,alpha,layered)[:self.N_units]
                xa_y_plus_2    = self.quantum_step_rfqrc(x_tilde, Yh_orig, Yh_plus,alpha,layered)[:self.N_units]
                xa_y_plus      = xa_y_plus_1+xa_y_plus_2

                xa_y_minus_1   = self.quantum_step_rfqrc(x_tilde, Yh_minus, Yh_orig,alpha,layered)[:self.N_units]
                xa_y_minus_2   = self.quantum_step_rfqrc(x_tilde, Yh_orig,Yh_minus,alpha,layered)[:self.N_units]
                xa_y_minus     = xa_y_minus_1 + xa_y_minus_2

            else:
                xa_y_plus    = self.quantum_step_rfqrc(x_tilde, Yh_plus,Yh_orig, alpha,layered)[:self.N_units]
                xa_y_minus   = self.quantum_step_rfqrc(x_tilde, Yh_minus,Yh_orig, alpha,layered)[:self.N_units]

            xa_y_grad[:,ll] = ( xa_y_plus - xa_y_minus ) * 0.5

        return xa_y_grad


    def quantum_gradient_calc_wrtx(self,xa,Yh,alpha):
        """_summary_

        Args:
            xa (_type_): _description_
            Yh (_type_): _description_
            alpha (_type_): _description_

        Returns:
            _xa_grad_: _Gradient wrt input parameter_
        """


        x_plus = xa.copy()
        x_plus = xa+(np.pi/2)

        x_minus = xa.copy()
        x_minus = xa-(np.pi/2)

        xa_x_plus    = self.quantum_step(x_plus, Yh, alpha)[:self.N_units]
        xa_x_minus   = self.quantum_step(x_minus, Yh, alpha)[:self.N_units]
        xa_x_grad = ( xa_x_plus - xa_x_minus ) * 0.5

        return xa_x_grad
    
    
    def quantum_open_loop_jacobian_leakrate(self,sys,N, dt, x0, alpha, Wout,L, norm_time):
        """ Advances ESN in open-loop and calculates the conditional lyapunov exponents.
            Args:
                N: number of time steps
                x0: initial reservoir state
                Wout: output matrix
            Returns:
                time series of prediction
                final augmented reservoir state
        """
        # Discard 1/10 of initial test set as a transient for the
        # tangent vectors to relax on the attractor.
        Ntransient = int(N/10)

        N_test = N - Ntransient
        Ttot  = np.arange(int(N_test/norm_time)) * dt * norm_time
    
        N_test_norm = int(N_test/norm_time)

        # Lyapunov Exponents timeseries
        LE = np.zeros((N_test_norm,self.dim))
        # finite-time Backward Lyapunov Exponents timeseries
        FTLE = np.zeros((N_test_norm,self.dim))
        # Q matrix recorded in time
        QQ_t = np.zeros((self.N_units,self.dim,N_test_norm))
        # R matrix recorded in time
        RR_t = np.zeros((self.dim,self.dim,N_test_norm))

        xa    = x0.copy()
        Yh    = np.empty((N, self.dim))
        xat   = np.empty((N, self.N_units))
        
        xat[0]= xa[:self.N_units]
        Yh[0] = np.dot(xa, Wout)

        #Initialize the GSVs
        U    = scipy.linalg.orth(np.random.rand(self.N_units,self.dim))
        Q, R = sys.qr_factorization(U)
        U    = Q[:,:self.dim].copy()
        #print('U_1',U.shape)
        
        xa_prev = xa[:self.N_units]
        # Yh_prev = U_in[0,:]
        
        for i in np.arange(1,Ntransient):
            #print('Step #',i)
            xa    = self.quantum_step(xa_prev, Yh[i-1], alpha)

            Yh[i] = np.dot(xa, Wout)
            xat[i]= xa[:self.N_units].copy()

            #0diag_mat = np.diag(self.quantum_grad(xa[:self.N_units],xa_prev[:self.N_units]))
            jacobian = (np.diag(np.ones(self.N_units) * self.epsilon_q)).dot(np.matmul(np.zeros((self.N_units,self.dim)), Wout[:self.N_units, :].T)) + (np.diag(np.ones(self.N_units) * (1 - self.epsilon_q)))
            U        = np.matmul(jacobian, U)

            #print('U_2',U.shape)
            if i % norm_time == 0:
                Q, R  = sys.qr_factorization(U)
                U    = Q[:,:self.dim].copy()

                
            xa_prev = xa[:self.N_units]
        
        
        xa_prev = xa[:self.N_units]
        indx = 0
        for i in np.arange(Ntransient,N):
            #print('U_2',U.shape)
            xa    = self.quantum_step(xa_prev, Yh[i-1], alpha)

            Yh[i] = np.dot(xa, Wout)
            xat[i]= xa[:self.N_units].copy()

            jacobian = (np.diag(np.ones(self.N_units) * self.epsilon_q)).dot(np.matmul(np.zeros((self.N_units,self.dim)), Wout[:self.N_units, :].T)) + (np.diag(np.ones(self.N_units) * (1 - self.epsilon_q)))
            U        = np.matmul(jacobian, U)
            #print(U.shape)
            if i % norm_time == 0:
                Q, R = sys.qr_factorization(U)
                U     = Q[:,:self.dim].copy()

                RR_t[:,:,indx] = R.copy()
                QQ_t[:,:,indx] = Q.copy()
                LE[indx]       = np.abs(np.diag(R[:self.dim,:self.dim]))
                FTLE[indx]     = (1./dt)*np.log(LE[indx])
                indx          +=1

                if i%20000==0:
                    print('Inside closed loop i=',i)

            xa_prev = xa[:self.N_units]

        LEs = np.cumsum(np.log(LE[1:]),axis=0) / np.tile(Ttot[1:],(self.dim,1)).T
        print('Conditional Lyapunov exponents: ',LEs[-1])

        return Yh, xa, LEs


    def quantum_closed_loop_jacobian_leakrate(self,sys,N, dt, x0, alpha, Wout, norm,norm_time,layered):
        """ Advances ESN in closed-loop and calculates the ESN Jacobian and Lyapunov exponents and vectors.
            Args:
                N: number of time steps
                x0: initial reservoir state
                Wout: output matrix
            Returns:
                time series of prediction
                final augmented reservoir state
        """
        # Discard 1/10 of initial test set as a transient for the
        # tangent vectors to relax on the attractor.
        Ntransient = int(N/10)

        N_test = N - Ntransient
        Ttot  = np.arange(int(N_test/norm_time)) * dt * norm_time

        N_test_norm = int(N_test/norm_time)

        # Lyapunov Exponents timeseries
        LE = np.zeros((N_test_norm,self.dim))
        # finite-time Backward Lyapunov Exponents timeseries
        FTLE = np.zeros((N_test_norm,self.dim))
        # Q matrix recorded in time
        QQ_t = np.zeros((self.N_units,self.dim,N_test_norm))
        # R matrix recorded in time
        RR_t = np.zeros((self.dim,self.dim,N_test_norm))

        xa    = x0.copy()
        Yh    = np.empty((N, self.dim))
        xat   = np.empty((N, self.N_units))
        xat[0]= xa[:self.N_units]
        Yh[0] = np.dot(xa, Wout)


        #Initialize the GSVs
        U    = scipy.linalg.orth(np.random.rand(self.N_units,self.dim))
        Q, R = sys.qr_factorization(U)
        U    = Q[:,:self.dim].copy()
        #print('U_1',U.shape)

        #const_jac_a, const_jac_b = self.const_jacobian(Win,W,Wout,norm) ####

        xa_prev = xa[:self.N_units]
        for i in np.arange(1,Ntransient):
            #print('Step #',i)
            xa    = self.quantum_step(xa_prev, Yh[i-1], alpha)

            xa_y_grad = self.quantum_gradient_calc_wrty(xa[:self.N_units],xa_prev[:self.N_units],Yh[i-1],alpha,layered)

            #xa_x_grad = self.quantum_gradient_calc_wrtx(xa[:self.N_units],xa_prev[:self.N_units],Yh[i-1],alpha)

            # ## Quantum gradient wrt x
            # xa_x_plus    = self.quantum_step(xa_prev+np.pi/2, Yh[i-1],alpha)
            # xa_x_minus   = self.quantum_step(xa_prev-np.pi/2, Yh[i-1],alpha)
            # xa_x_grad    = (xa_x_plus + xa_x_minus)* 0.5

            # xa    = xa[:-1]
            Yh[i] = np.dot(xa, Wout)
            xat[i]= xa[:self.N_units].copy()

            #0diag_mat = np.diag(self.quantum_grad(xa[:self.N_units],xa_prev[:self.N_units]))
            jacobian = (np.diag(np.ones(self.N_units)*self.epsilon_q)).dot(np.matmul(xa_y_grad,Wout[:self.N_units,:].T)) + (np.diag(np.ones(self.N_units)*(1-self.epsilon_q))) #+ np.multiply((np.diag(np.ones(self.N_units)*self.epsilon_q)),xa_x_grad) #+ np.matmul(diag_mat,const_jac_b) # only work with epsilon == 1
            U        = np.matmul(jacobian, U)
            #print('U_2',U.shape)
            if i % norm_time == 0:
                Q, R  = sys.qr_factorization(U)
                U    = Q[:,:self.dim].copy()


            xa_prev = xa[:self.N_units]

        xa_prev = xa[:self.N_units]
        indx = 0
        for i in np.arange(Ntransient,N):
            #print('U_2',U.shape)
            xa    = self.quantum_step(xa_prev, Yh[i-1], alpha)

            xa_y_grad = self.quantum_gradient_calc_wrty(xa[:self.N_units],xa_prev[:self.N_units],Yh[i-1],alpha,layered)

            # ## Quantum gradient wrt x
            # xa_x_plus    = self.quantum_step(xa_prev+np.pi/2, Yh[i-1],alpha)
            # xa_x_minus   = self.quantum_step(xa_prev-np.pi/2, Yh[i-1],alpha)
            # xa_x_grad    = (xa_x_plus + xa_x_minus)* 0.5
            # xa    = xa[:-1]

            Yh[i] = np.dot(xa, Wout)
            xat[i]= xa[:self.N_units].copy()

            #diag_mat = np.diag(self.dtanh(xa[:self.N_units],xa_prev[:self.N_units]))
            jacobian = (np.diag(np.ones(self.N_units)*self.epsilon_q)).dot(np.matmul(xa_y_grad,Wout[:self.N_units,:].T)) + (np.diag(np.ones(self.N_units)*(1-self.epsilon_q))) #+ np.matmul(diag_mat,const_jac_b) # only work with epsilon == 1

            # jacobian = (np.diag(np.ones(self.N_units)*self.epsilon_q)).dot(np.matmul(diag_mat,const_jac_a) + np.matmul(diag_mat,const_jac_b)) + (np.diag(np.ones(self.N_units)*(1-self.epsilon_q))) # only work with epsilon == 1
            U          = np.matmul(jacobian, U)
            #print(U.shape)
            if i % norm_time == 0:
                Q, R = sys.qr_factorization(U)
                U     = Q[:,:self.dim].copy()

                RR_t[:,:,indx] = R.copy()
                QQ_t[:,:,indx] = Q.copy()
                LE[indx]       = np.abs(np.diag(R[:self.dim,:self.dim]))
                FTLE[indx]     = (1./dt)*np.log(LE[indx])
                indx          +=1

                if i%20000==0:
                    print('Inside closed loop i=',i)

            xa_prev = xa[:self.N_units]

        LEs = np.cumsum(np.log(LE[1:]),axis=0) / np.tile(Ttot[1:],(self.dim,1)).T
        print('RF-QRC Lyapunov exponents: ',LEs[-1])

        return QQ_t, RR_t , Yh, xa, LEs
