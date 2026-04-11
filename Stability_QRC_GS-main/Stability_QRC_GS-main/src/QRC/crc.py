import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs as sparse_eigs
import scipy


class EchoStateNetwork:
    def __init__(self,tikh,sigma_in,rho,epsilon,bias_in,bias_out,N_units,dim,density):
        self.tikh     = tikh
        self.sigma_in = sigma_in
        self.rho      = rho
        self.epsilon  = epsilon
        self.bias_in  = bias_in
        self.bias_out = bias_out
        self.N_units  = N_units
        self.dim      = dim
        self.density  = density

    # can or cannot use property here, it provides encapsulation
    # @property
    # def tikh(self):
    #     return(self._tikh)

    @property
    def tikh(self):
        return (self._tikh)

    @tikh.setter
    def tikh (self,value):
        self._tikh = value

    @property
    def sigma_in(self):
        return (self._sigma_in)

    @sigma_in.setter
    def sigma_in (self,value):
        self._sigma_in = value

    @property
    def rho(self):
        return (self._rho)

    @rho.setter
    def rho (self,value):
        self._rho = value

    @property
    def epsilon(self):
        return (self._epsilon)

    @epsilon.setter
    def epsilon (self,value):
        self._epsilon = value


    @property
    def norm_u(self):
        return (self._norm_u)

    @norm_u.setter
    def norm_u (self,value):
        self._norm_u = value

    def gen_input_matrix(self,seed):
        rnd = np.random.RandomState(seed)
        #sparse syntax for the input and state matrices
        Win  = lil_matrix((self.N_units,self.dim+1))
        for j in range(self.N_units):
            Win[j,rnd.randint(0, self.dim+1)] = rnd.uniform(-1, 1) #only one element different from zero
        Win = Win.tocsr()

        return Win

    def gen_reservoir_matrix(self,seed):
        rnd = np.random.RandomState(seed)
        W = csr_matrix( #on average only connectivity elements different from zero
        rnd.uniform(-1, 1, (self.N_units, self.N_units)) * (rnd.rand(self.N_units, self.N_units) < (self.density)))

        spectral_radius = np.abs(sparse_eigs(W, k=1, which='LM', return_eigenvectors=False))[0]
        W = (1/spectral_radius)*W #scaled to have unitary spec radius

        return W


    def step(self,x_pre, u,Win,W):

        # input is normalized and input bias added

        u_augmented = np.hstack((u/self.norm_u, self.bias_in))

        # reservoir update
        x_post      = (1-self.epsilon)*(x_pre)+self.epsilon*(np.tanh(Win.dot(u_augmented*self.sigma_in) + W.dot(self.rho*x_pre) ))

        #x_post      = (1-self.epsilon)*(x_pre)+self.epsilon*np.tanh(np.dot(u_augmented*self.sigma_in, Win) + self.rho*np.dot(x_pre, W))

        x_augmented = np.hstack((x_post, self.bias_out))

        return x_augmented


    def open_loop(self,U, x0,Win,W):
        """ Advances ESN in open-loop.
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
            Xa[i] = self.step(Xa[i-1,:self.N_units], U[i-1],Win,W)

        return Xa

    def closed_loop(self, N, x0, Wout,Win,W):
        """ Advances ESN in closed-loop.
            Args:
                N: number of time steps
                x0: initial reservoir state
                Wout: output matrix
            Returns:
                time series of prediction
                final augmented reservoir state
        """
        xa    = x0.copy()
        Yh    = np.empty((N+1, self.dim))
        Yh[0] = np.dot(xa, Wout)
        for i in np.arange(1,N+1):
            xa    = self.step(xa[:self.N_units], Yh[i-1],Win,W)
            Yh[i] = np.dot(xa, Wout) #np.linalg.multi_dot([xa, Wout])

        return Yh, xa

    def train(self,U_washout, U_train, Y_train,Win,W):
        """ Trains ESN.
            Args:
                U_washout: washout input time series
                U_train: training input time series
                tikh: Tikhonov factor
            Returns:
                time series of augmented reservoir states
                optimal output matrix
        """

        LHS = 0
        RHS = 0

        N  = U_train[0].shape[0]
        Xa  = np.zeros((U_washout.shape[0], N+1, self.N_units+1))

        for i in range(U_washout.shape[0]):

            ## washout phase
            xf_washout = self.open_loop(U_washout[i], np.zeros(self.N_units),Win,W)[-1,:self.N_units]

            ## open-loop train phase
            Xa[i] = self.open_loop(U_train[i], xf_washout,Win,W)

            ## Ridge Regression
            LHS  += np.dot(Xa[i,1:].T, Xa[i,1:])
            RHS  += np.dot(Xa[i,1:].T, Y_train[i])

        #solve linear system for each Tikhonov parameter
        Wout = np.zeros((self.tikh.size, self.N_units+1,self.dim))
        for j in np.arange(self.tikh.size):
            Wout[j] = np.linalg.solve(LHS + self.tikh[j]*np.eye(self.N_units+1), RHS)

        return Xa, Wout, LHS, RHS




    def train_MC(self,U_washout, U_train, Y_train):
        """ Trains ESN.
            Args:
                U_washout: washout input time series
                U_train: training input time series
                Y_train: prediction for next time step to match U_train
                tikh: Tikhonov factor
            Returns:
                Wout: optimal output matrix
        """

        ## washout phase
        # taking last reservoir state of washout [-1] and removing bias :-1
        xf    = self.open_loop(U_washout, np.zeros(self.N_units))[-1,:-1]

        LHS   = 0
        RHS   = 0
        #N_len = (U_train.shape[0]-1)

        ## open-loop train phase
        Xa1 = self.open_loop(U_train, xf)[1:]
        xf  = Xa1[-1,:-1].copy()

        LHS += np.dot(Xa1.T, Xa1)
        RHS += np.dot(Xa1.T, Y_train)

        LHS.ravel()[::LHS.shape[1]+1] += self.tikh

        Wout = np.linalg.solve(LHS, RHS)

        return Wout


    ## Lyapunov exponent calculation from Giorgios implementation https://github.com/MagriLab/EchoStateNetwork/

    ## Only works without leaky integral
    def const_jacobian(self,Win,W,Wout,norm):
        dfdu = np.r_[np.diag(self.sigma_in/norm),[np.zeros(self.dim)]]

        #dfdu = self.epsilon * np.multiply(Win[:, : self.dim], 1.0 / norm[: self.dim])

        #                 self._dfdu_const = self.alpha * self.W_in[:, : self.N_dim].multiply(
        #             1.0 / self.norm_in[1][: self.N_dim]
        #print(dfdu.shape)
        d    = Win.dot(dfdu)
        c    = np.matmul(d,Wout[:self.N_units,:].T)

        return c, W.dot(np.diag(np.ones(self.N_units)*self.rho))


    def closed_loop_jacobian(self,sys,N, dt, x0, Win,W, Wout, norm,norm_time):
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

        const_jac_a, const_jac_b = self.const_jacobian(Win,W,Wout,norm) ####

        for i in np.arange(1,Ntransient):
            #print('Step #',i)
            xa    = self.step(xa[:self.N_units], Yh[i-1], Win,W)
            # xa    = xa[:-1]
            Yh[i] = np.dot(xa, Wout)
            xat[i]= xa[:self.N_units].copy()

            diag_mat = np.diag(1 - xa[:self.N_units]*xa[:self.N_units])
            jacobian = (np.ones(self.N_units)*(self.epsilon))*(np.matmul(diag_mat,const_jac_a) + np.matmul(diag_mat,const_jac_b)) + np.ones(self.N_units)*(1-self.epsilon) # only work with epsilon == 1
            U        = np.matmul(jacobian, U)
            #print('U_2',U.shape)
            if i % norm_time == 0:
                Q, R  = sys.qr_factorization(U)
                U    = Q[:,:self.dim].copy()


        indx = 0
        for i in np.arange(Ntransient,N):
            #print('U_2',U.shape)
            xa    = self.step(xa[:self.N_units], Yh[i-1], Win,W)
            # xa    = xa[:-1]
            Yh[i] = np.dot(xa, Wout)
            xat[i]= xa[:self.N_units].copy()

            diag_mat = np.diag(1 - xa[:self.N_units]*xa[:self.N_units])
            jacobian = (np.ones(self.N_units)*(self.epsilon))*(np.matmul(diag_mat,const_jac_a) + np.matmul(diag_mat,const_jac_b)) + np.ones(self.N_units)*(1-self.epsilon) # only work with epsilon == 1
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
        LEs = np.cumsum(np.log(LE[1:]),axis=0) / np.tile(Ttot[1:],(self.dim,1)).T
        print('ESN Lyapunov exponents: ',LEs[-1])


        #Calculation of CLVs
        # if check_clv == 1:
        #     print("Calculate CLVs")
        #     CLV_calculation(system,params,const_jac_a, const_jac_b,QQ_t,RR_t,dim,Ntransient,\
        #                     N_units,False,dt*norm_time,fname, Yh, xat, norm_time,\
        #                     saveclvs, LEs, FTLE, subspace_LEs_indeces)

        #avFTLES = np.average(FTLE,axis=0)
        #print("average FTLEs",avFTLES)

        return Yh, xa, LEs


    def closed_loop_jacobian_leakrate(self,sys,N, dt, x0, Win,W, Wout, norm,norm_time):
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

        const_jac_a, const_jac_b = self.const_jacobian(Win,W,Wout,norm) ####

        xa_prev = xa[:self.N_units]
        for i in np.arange(1,Ntransient):
            #print('Step #',i)
            xa    = self.step(xa_prev, Yh[i-1], Win,W)
            # xa    = xa[:-1]
            Yh[i] = np.dot(xa, Wout)
            xat[i]= xa[:self.N_units].copy()

            diag_mat = np.diag(self.dtanh(xa[:self.N_units],xa_prev[:self.N_units]))
            jacobian = (np.diag(np.ones(self.N_units)*self.epsilon)).dot(np.matmul(diag_mat,const_jac_a) + np.matmul(diag_mat,const_jac_b)) + (np.diag(np.ones(self.N_units)*(1-self.epsilon)))
            #print('U_2',U.shape)
            if i % norm_time == 0:
                Q, R  = sys.qr_factorization(U)
                U    = Q[:,:self.dim].copy()


            xa_prev = xa[:self.N_units]

        xa_prev = xa[:self.N_units]
        indx = 0
        for i in np.arange(Ntransient,N):
            #print('U_2',U.shape)
            xa    = self.step(xa_prev, Yh[i-1], Win,W)
            # xa    = xa[:-1]
            Yh[i] = np.dot(xa, Wout)
            xat[i]= xa[:self.N_units].copy()

            diag_mat = np.diag(self.dtanh(xa[:self.N_units],xa_prev[:self.N_units]))
            jacobian = (np.diag(np.ones(self.N_units)*self.epsilon)).dot(np.matmul(diag_mat,const_jac_a) + np.matmul(diag_mat,const_jac_b)) + (np.diag(np.ones(self.N_units)*(1-self.epsilon)))
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
        print('ESN Lyapunov exponents: ',LEs[-1])

        return Yh, xa, LEs

    def dtanh(self, x, x_prev):
        """Derivative of the tanh part
        This derivative appears in different gradient calculations
        So, it makes sense to calculate it only once and call as required

        Args:
        x: reservoir states at time i+1, x(i+1)
        x_prev: reservoir states at time i, x(i)
        """
        # first we find tanh(...)
        #print(x.shape,x_prev.shape)
        x_tilde = (x - (1 - self.epsilon) * x_prev) / self.epsilon
        #print(x_tilde.shape)
        # derivative of tanh(...) is 1-tanh**2(...)
        dtanh = 1.0 - x_tilde**2

        return dtanh


    # ## Lyapunov exponent calculation from Elise and Defne's implementation

    # def closed_loop_states(self, N, x0, Wout,Win,W):
    #     """ Advances ESN in closed-loop.
    #         Args:
    #             N: number of time steps
    #             x0: initial reservoir state
    #             Wout: output matrix
    #         Returns:
    #             time series of prediction
    #             final augmented reservoir state
    #     """

    #     xa    = x0.copy()
    #     Yh    = np.empty((N+1, self.dim))
    #     Yh[0] = np.dot(xa, Wout)
    #     Xa    = np.empty((N+1,self.N_units))
    #     Xa[0] = xa[:self.N_units]

    #     for i in np.arange(1,N+1):
    #         xa    = self.step(xa[:self.N_units], Yh[i-1],Win,W)
    #         Yh[i] = np.dot(xa, Wout) #np.linalg.multi_dot([xa, Wout])
    #         Xa[i] = xa[:self.N_units]

    #     return Yh, Xa


    # def dtanh(self, x, x_prev):
    #     """Derivative of the tanh part
    #     This derivative appears in different gradient calculations
    #     So, it makes sense to calculate it only once and call as required

    #     Args:
    #     x: reservoir states at time i+1, x(i+1)
    #     x_prev: reservoir states at time i, x(i)
    #     """
    #     # first we find tanh(...)
    #     #print(x.shape,x_prev.shape)
    #     x_tilde = (x - (1 - self.epsilon) * x_prev) / self.epsilon
    #     #print(x_tilde.shape)
    #     # derivative of tanh(...) is 1-tanh**2(...)
    #     dtanh = 1.0 - x_tilde**2

    #     return dtanh



    # def jac(self, dtanh, x_prev=None):
    #     """Jacobian of the reservoir states, ESN in closed loop
    #     taken from
    #     Georgios Margazoglou, Luca Magri:
    #     Stability analysis of chaotic systems from data, arXiv preprint arXiv:2210.06167

    #     x(i+1) = f(x(i),u(i),p)
    #     df(x(i),u(i))/dx(i) = \partial f/\partial x(i) + \partial f/\partial u(i)*\partial u(i)/\partial x(i)

    #     x(i+1) = (1-alpha)*x(i)+alpha*tanh(W_in*[u(i);p]+W*x(i))

    #     Args:
    #     dtanh: derivative of tanh at x(i+1), x(i)

    #     Returns:
    #     dfdx: jacobian of the reservoir states
    #     """

    #     # gradient of x(i+1) with x(i) due to x(i) that appears explicitly
    #     # no reservoir connections
    #     dfdx_x = self.dfdx_x_const
    #     #print(self.W_le.shape,dtanh.shape)
    #     dfdx_x += self.epsilon * self.W_le.multiply(dtanh)
    #     # gradient of x(i+1) with x(i) due to u(i) (in closed-loop)
    #     dfdx_u = self.dfdx_u(dtanh, x_prev)
    #     # total derivative
    #     dfdx = dfdx_x + dfdx_u
    #     return dfdx



    # @property
    # def dfdx_x_const(self):
    #     # constant part of gradient of x(i+1) with respect to x(i) due to x(i)
    #     # sparse matrix
    #     if not hasattr(self, "_dfdx_x_const"):
    #         self._dfdx_x_const = csr_matrix((1 - self.epsilon) * np.eye(self.N_units))
    #     return self._dfdx_x_const


    # @property
    # def dfdx_u(self):
    #     if not hasattr(self, "_dfdx_u"):
    #         self._dfdx_u = self.dfdx_u_r1
    #     return self._dfdx_u


    # def dfdx_u_r1(self, dtanh, x_prev=None):
    #     return np.multiply(self.dfdu_dudx_const, dtanh)


    # # def dfdx_u_r2(self, dtanh, x_prev=None):
    # #     # derivative of x**2 terms
    # #     dx_prev = np.ones(self.N_units)
    # #     dx_prev[1::2] = 2 * x_prev[1::2]

    # #     dudx = np.multiply(self.dudx_const, dx_prev)
    # #     dfdu_dudx = self.dfdu_const.dot(dudx)
    # #     return np.multiply(dfdu_dudx, dtanh)

    # @property
    # def dfdu_dudx_const(self):
    #     # constant part of gradient of x(i+1) with respect to x(i) due to u_in(i)
    #     # not sparse matrix
    #     if not hasattr(self, "_dfdu_dudx_const"):
    #         self._dfdu_dudx_const = self.dfdu_const.dot(self.dudx_const)
    #     return self._dfdu_dudx_const


    # @property
    # def dfdu_const(self):
    #     # constant part of gradient of x(i+1) with respect to u_in(i)
    #     # sparse matrix
    #     if not hasattr(self, "_dfdu_const"):
    #         #try:
    #         self._dfdu_const = self.epsilon * self.W_in_le[:, : self.dim].multiply(
    #             1.0 / self.norm_u[: self.dim])
    #         # except:
    #         #     self._dfdu_const = self.epsilon * np.multiply(
    #         #         self.W_in_le[:, : self.dim], 1.0 / self.norm_u[: self.dim]
    #         #     )
    #     return self._dfdu_const

    # @property
    # def dudx_const(self):
    #     # gradient of u_in(i) with respect to x(i)
    #     # not sparse matrix
    #     return self.W_out_le[: self.N_units, :].T

    # @property
    # def W_le(self):
    #     return (self._W_le)

    # @W_le.setter
    # def W_le (self,value):
    #     self._W_le = value

    # @property
    # def W_in_le(self):
    #     return (self._W_in_le)

    # @W_in_le.setter
    # def W_in_le (self,value):
    #     self._W_in_le = value

    # @property
    # def W_out_le(self):
    #     return (self._W_out_le)

    # @W_out_le.setter
    # def W_out_le(self,value):
    #     self._W_out_le = value
