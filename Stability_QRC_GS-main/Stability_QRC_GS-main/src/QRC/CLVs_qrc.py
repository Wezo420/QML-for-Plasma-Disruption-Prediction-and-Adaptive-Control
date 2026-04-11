import numpy as np
import scipy
import itertools
import math
import h5py

def normalize(M):
    ''' Normalizes columns of M individually '''
    nM = np.zeros(M.shape) # normalized matrix
    nV = np.zeros(np.shape(M)[1]) # norms of columns

    for i in range(M.shape[1]):
        nV[i] = scipy.linalg.norm(M[:,i])
        nM[:,i] = M[:,i] / nV[i]

    return nM, nV

def timeseriesdot(x,y, multype): 
    tsdot = np.einsum(multype,x,y.T) #Einstein summation. Index i is time.
    return tsdot

def vec_normalize(vec,timeaxis):
    #normalize a vector within its timeseries
    timetot = np.shape(vec)[timeaxis]
    for t in range(timetot):        
        vec[t,:] = vec[t,:] / np.linalg.norm(vec[t,:])
    return vec

def subspace_angles(clv, timeaxis, index):
    #calculate principal angles between subspaces    
    timetot = np.shape(clv)[timeaxis]      
    thetas = np.zeros((timetot,3))
    
    #Nv_un and clvs of the unstable expanding subspace
    Nv_un = index[0]
    CLV_un = clv[:,0:Nv_un,:]
    print('CLV_un shape', CLV_un.shape)
    pos_clvs = Nv_un
    
    #Nv_nu and clvs of the neutral subspace
    Nv_nu = index[1]
    CLV_nu = clv[:,Nv_un:Nv_un+Nv_nu,:]
    print('CLV_nu shape', CLV_nu.shape)
    neut_clvs = Nv_nu
    
    #clvs of the stable decaying subspace
    CLV_st = clv[:,Nv_un+Nv_nu:,:]
    print('CLV_st shape', CLV_st.shape)
    neg_clvs = np.shape(CLV_st)[1]
    
    for t in range(timetot):        
        thetas[t,0] = np.rad2deg(scipy.linalg.subspace_angles(CLV_un[:,:,t],CLV_nu[:,:,t]))[-1]
        thetas[t,1] = np.rad2deg(scipy.linalg.subspace_angles(CLV_un[:,:,t],CLV_st[:,:,t]))[-1]
        thetas[t,2] = np.rad2deg(scipy.linalg.subspace_angles(CLV_nu[:,:,t],CLV_st[:,:,t]))[-1]
    
    return thetas

def CLV_angles(clv, NLy):
    #calculate angles between CLVs        
    costhetas = np.zeros((clv[:,0,:].shape[1],NLy))
    count = 0
    for subset in itertools.combinations(np.arange(NLy), NLy-1):
        index1 = subset[0]
        index2 = subset[1]        
        #For principal angles take the absolute of the dot product        
        costhetas[:,count] = np.absolute(timeseriesdot(clv[:,index1,:],clv[:,index2,:],'ij,ji->j'))
        count+=1
    thetas = 180. * np.arccos(costhetas) / math.pi
    
    return thetas

def CLV_calculation(system,QQ,RR,NLy,Ntherm,U_dim,dt,fname,state, norm_time, saveclvs, LEs, FTLE, subspace_LEs_indeces):
    
    tly = np.shape(QQ)[-1]
    su = int(tly / 10)
    sd = int(tly / 10)
    s  = su              # index of spinup time
    e  = tly+1 - sd      # index of spindown time
    tau = int(dt/dt)     # time for finite-time lyapunov exponents

    #Calculation of CLVs
    C = np.zeros((NLy,NLy,tly))        # coordinates of CLVs in local GS vector basis
    D = np.zeros((NLy,tly))            # diagonal matrix with CLV growth factors
    V = np.zeros((U_dim,NLy,tly))      # coordinates of CLVs in physical space (each column is a vector)


    # FTCLE: Finite-time lyapunov exponents along CLVs
    il  = np.zeros((NLy,tly+1)) 

    # Dynamical system
    # if system=='lorenz96':
    #     eom = lorenz96
    #     jac = l96jac
    # if system=='lorenz63':
    #     eom = lorenz63
    #     jac = l63jac
    # if system=='rossler':
    #     eom = rossler
    #     jac = rosjac
    # if system=='cdv':
    #     eom = cdv
    #     jac = cdvjac

    # initialise components to I
    C[:,:,-1] = np.eye(NLy)
    D[:,-1]   = np.ones(NLy)
    V[:,:,-1] = np.dot(np.real(QQ[:,:,-1]), C[:,:,-1])

    for i in reversed(range( tly-1 ) ):
        C[:,:,i], D[:,i] = normalize(scipy.linalg.solve_triangular(np.real(RR[:,:,i]), C[:,:,i+1]))
        V[:,:,i]         = np.dot(np.real(QQ[:,:,i]), C[:,:,i])

    
    # Computes the FTCLEs
    for j in 1+np.arange(s, e): #time loop
        il[:,j] = -(1./dt)*np.log(D[:,j])
        

    hf = h5py.File(fname+'.h5','w')
    if saveclvs == 'all':
        hf.create_dataset('CLV',      data=V)
        hf.create_dataset('FTCLE',    data=il)
        # hf.create_dataset('FTLE',     data=FTLE)
        hf.create_dataset('LEs',      data=LEs)
        hf.create_dataset('state',    data=state)        
        hf.create_dataset('Ntherm',   data=Ntherm)
        hf.create_dataset('norm_time',data=norm_time)
    elif saveclvs == 'angles':
        #normalize CLVs before measuring their angles.
        timetot = np.shape(V)[-1]
        for i in range(NLy):
            for t in range(timetot-1):
                V[:,i,t] = V[:,i,t] / np.linalg.norm(V[:,i,t])

        if system == 'VPT_L96_{}'.format(NLy):  
            thetas_clv = subspace_angles(V, -1, subspace_LEs_indeces)
        else:
            thetas_clv = CLV_angles(V, NLy)        

        
    #     hf.create_dataset('thetas_clv', data=thetas_clv)        
    #     hf.create_dataset('FTCLE',      data=il)
    #     hf.create_dataset('FTLE',       data=FTLE)
    #     hf.create_dataset('LEs',        data=LEs)
    #     hf.create_dataset('state',      data=state)
    #     hf.create_dataset('Ntherm',     data=Ntherm)
    #     hf.create_dataset('norm_time',  data=norm_time)

    
    # hf.close()
    
    return V , il , FTLE , LEs, thetas_clv 


# Code from https://github.com/MagriLab/LatentStability/blob/main/stabilitytools/clvs.py

def normalize(M):
    """Normalizes columns of M individually"""
    nM = np.zeros(M.shape)  # normalized matrix
    nV = np.zeros(np.shape(M)[1])  # norms of columns

    for i in range(M.shape[1]):
        nV[i] = scipy.linalg.norm(M[:, i])
        nM[:, i] = M[:, i] / nV[i]

    return nM, nV


def timeseriesdot(x, y, multype):
    tsdot = np.einsum(multype, x, y.T)  # Einstein summation. Index i is time.
    return tsdot


def compute_CLV(QQ, RR, dt):
    """
    Computes the Covariant Lyapunov Vectors (CLVs) using the method of Ginelli et al. (PRL, 2007).

    Parameters
    ----------
    QQ : numpy.ndarray
        Array of shape (N_cells, NLy, tly) containing the timeseries of Gram-Schmidt vectors.
    RR : numpy.ndarray
        Array of shape (NLy, NLy, tly) containing the timeseries of the upper-triangular R matrices from QR decomposition.
    dt : float
        Integration time step for the system.

    Returns
    -------
    V : numpy.ndarray
        Array of shape (N_cells, NLy, tly) containing the Covariant Lyapunov Vectors (CLVs) for each timestep. Each column represents a CLV in physical space.

    Notes
    -----
    - The CLVs are computed in reverse time by iteratively solving triangular systems from the QR decomposition.
    - The method normalizes the vectors at each timestep to avoid numerical instability.
    - Ginelli et al. (PRL, 2007) provides the theoretical foundation for this algorithm.
    """
    n_cells_x2 = QQ.shape[0]
    NLy = QQ.shape[1]
    tly = np.shape(QQ)[-1]

    # coordinates of CLVs in local GS vector basis
    C = np.zeros((NLy, NLy, tly))
    D = np.zeros((NLy, tly))  # diagonal matrix
    # coordinates of CLVs in physical space (each column is a vector)
    V = np.zeros((n_cells_x2, NLy, tly))

    # initialise components to I
    C[:, :, -1] = np.eye(NLy)
    D[:, -1] = np.ones(NLy)
    V[:, :, -1] = np.dot(np.real(QQ[:, :, -1]), C[:, :, -1])

    for i in reversed(range(tly-1)):
        C[:, :, i], D[:, i] = normalize(
            scipy.linalg.solve_triangular(np.real(RR[:, :, i]), C[:, :, i+1]))
        V[:, :, i] = np.dot(np.real(QQ[:, :, i]), C[:, :, i])

    # normalize CLVs before measuring their angles.
    timetot = np.shape(V)[-1]

    for i in range(NLy):
        for t in range(timetot-1):
            V[:, i, t] = V[:, i, t] / np.linalg.norm(V[:, i, t])
    return V


def compute_thetas(V, clv_idx):
    """
    Compute the cosines and angles (in degrees) between subspaces
    defined by the vectors in V for given CLV index pairs.

    Parameters
    ----------
    V : ndarray
        Array of shape (timesteps, subspace_dim, vectors) representing the CLVs (Covariant Lyapunov Vectors).
    clv_idx : list of tuples
        List of index pairs (i, j) indicating which CLV vectors to compare.

    Returns
    -------
    costhetas : ndarray
        Cosines of the angles between the specified CLV vector pairs.
    thetas : ndarray
        Angles (in degrees) between the specified CLV vector pairs.
    """
    costhetas = np.array([np.abs(timeseriesdot(V[:, i, :], V[:, j, :], 'ij,ji->j')) for i, j in clv_idx]).T
    thetas = np.degrees(np.arccos(costhetas))
    return costhetas, thetas