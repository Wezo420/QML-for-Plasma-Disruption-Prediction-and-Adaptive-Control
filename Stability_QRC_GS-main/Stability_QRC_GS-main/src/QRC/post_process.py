import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib as mpl
import numpy as np
import warnings
import scipy.stats as stats 


def plot_lorenz63_CLV_stats_predq(clv,clv_pred,dist):
    
    label = ['{U,N}','{U,S}','{N,S}']
    dim = clv.shape[1]
    # len_signal=len(Ys_t)
 
    fig, axs = plt.subplots(1,3)
    # fig.set_figwidth(18)
    # fig.set_figheight(6)

    for i in range(dim):
        signal = clv[:,i]
        signal_p = clv_pred[:,i]
        
        xmin, xmax = np.min(clv[:,i]), np.max(clv[:,i])
        print(xmin,xmax)
        x = np.linspace(xmin,xmax,dist)
        
        xminp, xmaxp = np.min(clv_pred[:,i]), np.max(clv_pred[:,i])
        print(xminp,xmaxp)
        xp = np.linspace(xminp,xmaxp,dist)
        
        kde = stats.gaussian_kde(signal)
        kde_p = stats.gaussian_kde(signal_p)

        axs[i].plot(x,kde(x),label='Ground truth',color='gray',linewidth='4',alpha=0.75)
        axs[i].plot(xp,kde_p(xp),label='RF-QRC',color='Red',linewidth='4',alpha=.8,linestyle='--')
        pos = axs[i].get_position()  # Get current position
        axs[i].set_position([pos.x0 + 0.1, pos.y0, pos.width * 0.90, pos.height])  # Reduce width

        # axs[i].set_box_aspect(0.5)
        axs[i].set_yscale("log")
        # axs[i].plot(x,kde_p(x),label='CRCM',color='red',linewidth='1.5',linestyle='--')
        # if quantum:
        #     axs[i].plot(x,kde_p_q(x),label='QRCM',color='black',linewidth='0.9')
        if i ==0:
            axs[i].set_ylabel('PDF')
        axs[i].set_xlabel(r'$\theta_{}$'.format(label[i]))
        axs[i].grid(alpha=0.2)
    
    
    return kde , kde_p


def plot_lorenz63_attractor_clvs_subplot_scaled(U,length,c_angle,color_plot):
    """A function to plot input data of Lorenz 63

    Args:
        U (array): Input Time Series
        length (scalar): length of Plot 1

    Return:
        Plot 1 : 3d Plot of x
        Plot 2 : Time Series wrt Number of Steps
        Plot 3 : Time Series wrt Lyapunov Time
        Plot 4 : Convection Current and Thermal Plots
    """

    # 3D PLOT OF LORENZ 63 ATTRACTOR

    label = ['{U,N}','{U,S}','{N,S}']
    dim   = U.shape[1]
    # fig = plt.figure()
    # fig.set_figheight(5)
    # fig.set_figwidth(16)
    
    fig, axs = plt.subplots(1, dim, subplot_kw={'projection': '3d'}, figsize=(26, 16))

    # Store the scatter plots in a list
    scatters = []

    for i, ax in enumerate(axs):

        # ax.set_xticks([])  # Turn off x-axis numbers
        # ax.set_yticks([])  # Turn off y-axis numbers
        # ax.set_zticks([])  # Turn off z-axis numbers
        ax.set_xlabel("$x_{1}$",labelpad=23)
        ax.set_ylabel("$x_{2}$",labelpad=23)
        ax.set_zlabel("$x_{3}$",labelpad=30)
        ax.set_zlim([0,1])
        ax.set_title(r'$\theta_{}$'.format(label[i]))
        
        ax.tick_params(axis='x', pad=10)  # Increase horizontal padding
        ax.tick_params(axis='y', pad=10)  # Increase vertical padding
        ax.tick_params(axis='z', pad=15)  # Increase vertical padding
        
        # pos = ax.get_position()  # Get current position
        # ax.set_position([pos.x0 + 0.1, pos.y0, pos.width * 2.6, pos.height])  # Reduce width
        #Plotting the 3D scatter on each subplot
        scatter = ax.scatter(*U[:length,:].T, lw=0.4, c = c_angle[:,i] , cmap=color_plot,marker='x')
        scatters.append(scatter)
        ax.grid("True")
    
    # Adjust the layout to prevent the right subplot from being cut off
    # plt.subplots_adjust(left=0.1, right=0.90, top=0.9, bottom=0.1, wspace=0.1)
    # Adding a single horizontal colorbar below all subplots
    #cbar = fig.colorbar(scatters[0], ax=axs, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar = fig.colorbar(scatters[0], ax=axs, orientation='horizontal', fraction=0.07, pad=0.15, 
                    location='bottom', use_gridspec=False)
    cbar.set_label('CLV angles')
    # plt.show()


def plot_lorenz63_CLV_stats(clv,dist):
    
    label = ['{U,N}','{U,S}','{N,S}']
    dim = clv.shape[1]
    # len_signal=len(Ys_t)
    #
    
    plt.rcParams['text.usetex'] = True
    # plt.rcParams["figure.figsize"] = (10,6)
    plt.rcParams["font.size"] = 15

    fig, axs = plt.subplots(1, dim, figsize=(12, 4))
    # fig.set_figheight(5)
    # fig.set_figwidth(16)
    


    for i in range(dim):
        signal = clv[:,i]
        xmin, xmax = np.min(clv[:,i]), np.max(clv[:,i])
        print(xmin,xmax)
        x = np.linspace(xmin,xmax,dist)
        # signal_p  = Ys_t[-len_signal:,i] #predicted closed loop
        
        kde = stats.gaussian_kde(signal)

        axs[i].plot(x,kde(x),label='True',color='gray',linewidth='4',alpha=0.75)
        axs[i].set_yscale("log")
        if i ==0:
            axs[i].set_ylabel('PDF')
        axs[i].set_xlabel(r'$\theta_{}$'.format(label[i]))
        axs[i].grid(alpha=0.2)
            

    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))            
    # plt.legend()
    # plt.yscale("log")
    # plt.show()    

    return kde



def plot_lorenz63_attractor_clvs_subplot(U,length,c_angle,color_plot):
    """A function to plot input data of Lorenz 63

    Args:
        U (array): Input Time Series
        length (scalar): length of Plot 1

    Return:
        Plot 1 : 3d Plot of x
        Plot 2 : Time Series wrt Number of Steps
        Plot 3 : Time Series wrt Lyapunov Time
        Plot 4 : Convection Current and Thermal Plots
    """

    # 3D PLOT OF LORENZ 63 ATTRACTOR

    label = ['{U,N}','{U,S}','{N,S}']
    dim   = U.shape[1]
    # fig = plt.figure()
    # fig.set_figheight(5)
    # fig.set_figwidth(16)
    
    fig, axs = plt.subplots(1, dim, subplot_kw={'projection': '3d'}, figsize=(26, 16))

    # Store the scatter plots in a list
    scatters = []

    for i, ax in enumerate(axs):

        # ax.set_xticks([])  # Turn off x-axis numbers
        # ax.set_yticks([])  # Turn off y-axis numbers
        # ax.set_zticks([])  # Turn off z-axis numbers
        ax.set_xlabel("$x_{1}$",labelpad=23)
        ax.set_ylabel("$x_{2}$",labelpad=23)
        ax.set_zlabel("$x_{3}$",labelpad=30)
        # ax.set_zlim([0,1])
        ax.set_title(r'$\theta_{}$'.format(label[i]))
        
        ax.tick_params(axis='x', pad=10)  # Increase horizontal padding
        ax.tick_params(axis='y', pad=10)  # Increase vertical padding
        ax.tick_params(axis='z', pad=15)  # Increase vertical padding
        
        # pos = ax.get_position()  # Get current position
        # ax.set_position([pos.x0 + 0.1, pos.y0, pos.width * 2.6, pos.height])  # Reduce width
        #Plotting the 3D scatter on each subplot
        scatter = ax.scatter(*U[:length,:].T, lw=0.4, c = c_angle[:,i] , cmap=color_plot,marker='x')
        scatters.append(scatter)
        ax.grid("True")
    
    # Adjust the layout to prevent the right subplot from being cut off
    # plt.subplots_adjust(left=0.1, right=0.90, top=0.9, bottom=0.1, wspace=0.1)
    # Adding a single horizontal colorbar below all subplots
    #cbar = fig.colorbar(scatters[0], ax=axs, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar = fig.colorbar(scatters[0], ax=axs, orientation='horizontal', fraction=0.07, pad=0.15, 
                    location='bottom', use_gridspec=False)
    cbar.set_label('CLVs')
    # plt.show()


def last_positive_cumulative_sum(arr):
    total_sum = 0
    last_positive_sum = 0  # Store the last positive sum
    for i in range(len(arr)):
        total_sum += arr[i]
        if total_sum < 0:
            return last_positive_sum , i # i here not i-1 because array index is 0 and we have index starting from 1
        last_positive_sum = total_sum  # Update when the sum is still positive
    return last_positive_sum  # In case the sum never becomes negative


def ky_dimension(arr):
    numerator, kk = last_positive_cumulative_sum(arr)
    ky = kk + numerator / (np.abs(arr[kk]))
    return ky

def calculate_snr(clean_signal, noisy_signal):
    # Compute signal component (difference between clean and noisy signals)
    noise = noisy_signal - clean_signal

    # Compute noise component (standard deviation of noisy signal)
    noise = np.mean(noisy_signal)

    # Calculate SNR
    snr = np.mean(clean_signal) / noise

    return snr

def signaltonoise(a, b, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    c = np.abs(b-a)
    if c.all() == 0:
        warnings.warn("Non-noisy signal, the SNR is inf ")

    sd = c.std(axis=axis, ddof=ddof)

    return np.where(sd == 0, 0, m/sd)


def add_noise(U,target_snr_db,seed,dim,N0,N1):
    """_summary_

    Args:
        U (_flattened_array_): _2dim_Input_Data(Flattened)

    Returns:
        _UU_: _3dim_Reshaped_array
    """
    #### adding noise component-wise to the data

    # Set a target SNR in decibel
    #target_snr_db = 40
    sig_avg_watts = np.var(U,axis=0) #signal power
    sig_avg_db = 10 * np.log10(sig_avg_watts) #convert in decibel
    # Calculate noise then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.zeros(U.shape)
    seed = 0                        #to be able to recreate the data
    rnd  = np.random.RandomState(seed)
    for i in range(dim):
        noise_volts[:,i] = rnd.normal(mean_noise, np.sqrt(noise_avg_watts[i]),
                                        U.shape[0])
    UU  = U + noise_volts
    UU  = UU.reshape(N0,N1,dim)

    return UU

def plot_lorenz63_attractor(U,length):
    """A function to plot input data of Lorenz 63

    Args:
        U (array): Input Time Series
        length (scalar): length of Plot 1

    Return:
        Plot 1 : 3d Plot of x
        Plot 2 : Time Series wrt Number of Steps
        Plot 3 : Time Series wrt Lyapunov Time
        Plot 4 : Convection Current and Thermal Plots
    """

    # 3D PLOT OF LORENZ 63 ATTRACTOR

    plt.rcParams['text.usetex'] = True
    plt.rcParams["figure.figsize"] = (10,6)
    plt.rcParams["font.size"] = 12
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlabel("$x_{1}$",labelpad=5)
    ax.set_ylabel("$x_{2}$",labelpad=5)
    ax.set_zlabel("$x_{3}$",labelpad=5)

    plt.tight_layout()

    ax.plot(*U[:length,:].T, lw=0.6, c ='blue')
    #ax.scatter(*U[10:11,:].T, c ='red')
    ax.dist = 11.5
    plt.legend(['Training Data'])


def plot_lorenz63_time(U,N_lyap,l2,l3):
    """A function to plot input data of Lorenz 63

    Args:
        U (array): Input Time Series
        l1 (scalar): length of Plot 1
        l2 (scalar): length of Plot 2
        l3 (scalar): length of Plot 3

    Return:
        Plot 1 : 3d Plot of Attractor
        Plot 2 : Time Series wrt Number of Steps
        Plot 3 : Time Series wrt Lyapunov Time
        Plot 4 : Convection Current and Thermal Plots
    """

    # PLOTTING TIME EVOLUTION OF A1,B1,B2 in time steps

    t_len = l2 # length of time series to plot
    t_str = 0 # starting points for plots
    #### 222 time steps in one Lyapunov Time, have to do it by 3Lambda to replicate the paper
    fig, axs = plt.subplots(3)
    fig.suptitle('Time steps Vs $x_{1}$,$x_{2}$,$x_{3}$')
    axs[0].plot(np.arange(t_str,t_len),U[t_str:t_len,0])
    axs[0].set_ylabel("$x_{1}$")
    axs[1].plot(np.arange(t_str,t_len),U[t_str:t_len,1])
    axs[1].set_ylabel("$x_{2}$")
    axs[2].plot(np.arange(t_str, t_len),U[t_str:t_len,2])
    axs[2].set_xlabel('Time steps')
    axs[2].set_ylabel("$x_{3}$")

    # PLOTTING TIME EVOLUTION OF A1,B1,B2 in LYAP TIME
    # Readjusting Lyap Time to zero

    t_len = l3 # length of time series to plot
    t_str = 0 # starting points for plots, because have removed transients already
    #### 222 time steps in one Lyapunov Time, have to do it by 3Lambda to replicate the paper
    fig, axs = plt.subplots(3)
    fig.suptitle('Lyapunov Time Vs $x_{1}$,$x_{2}$,$x_{3}$')
    axs[0].plot(np.arange(t_str,t_len)/N_lyap,U[t_str:t_len,0])
    axs[0].set_ylabel("$x_{1}$")
    axs[1].plot(np.arange(t_str,t_len)/N_lyap,U[t_str:t_len,1])
    axs[1].set_ylabel("$x_{2}$")
    axs[2].plot(np.arange(t_str, t_len)/N_lyap,U[t_str:t_len,2])
    axs[2].set_xlabel('Lyapunov Time (LT)')
    axs[2].set_ylabel("$x_{3}$")


def lorenz63_timeseries_plot(N_test,N_tstart,N_intt,N_fwd,j,UU,U,U_test,N_washout,N_lyap,ensemble,N_t,minimum,esn,Win,W,QESN,Woutt,Woutt_qq,alpha_qq,plot=True,quantum=True):

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(UU)))
    threshold_ph = 0.5
    num_series = 0
    n_plot = 3

    PH_plot   = np.zeros((ensemble,3))
    PH_plot_q = np.zeros((ensemble,3))

    print(U.shape, UU.shape, N_t)


    print('Realization    :',j+1)
    #load matrices and hyperparameters
    Wout     = Woutt[j]
    # Win      = Winn[j]
    # W        = Ws[j]
    esn.rho      = 10**minimum[j,0]
    esn.epsilon  = minimum[j,1]
    esn.sigma_in = minimum[j,2]

    if quantum:
        Wout_q             = Woutt_qq[j]
        alpha              = alpha_qq[j]

        print('Hyperparameters:',esn.rho, esn.epsilon,esn.sigma_in,esn.tikh)
        print('Quantum Hyperparameters:', QESN.rho_q, QESN.epsilon_q,QESN.sigma_in_q,QESN.tikh_q,'Seed:',j+1)

        # to store prediction horizon in the test set
        PH         = np.zeros(N_test)
        PH_q       = np.zeros(N_test)
        # to plot results

        if plot:
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout(h_pad=12)

        #run different test intervals
        for i in range(N_test):

            # data for washout and target in each interval
            U_wash    = U_test[num_series,N_tstart - N_washout +i*N_fwd : N_tstart + i*N_fwd].copy()
            Y_t       = U_test[num_series,N_tstart  +i*N_fwd           : N_tstart + i*N_fwd + N_intt].copy()

            #washout for each interval
            Xa1     = esn.open_loop(U_wash, np.zeros(esn.N_units),Win,W)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t        = esn.closed_loop(N_intt-1, Xa1[-1], Wout,Win,W)[0]
            Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph:
                PH[i] = N_intt/N_lyap #(in case PH is larger than interval)


            if quantum:
                #washout for each interval
                Xa1_q   = QESN.quantum_openloop(U_wash, np.zeros(QESN.N_units),alpha) # here concatenation and bias out addition

                Uh_wash_q = np.dot(Xa1_q, Wout_q)

                # Prediction Horizon
                Yh_t_q        = QESN.quantum_closedloop(N_intt-1, Xa1_q[-1], Wout_q,alpha)[0]
                Y_err_q       = np.sqrt(np.mean((Y_t-Yh_t_q)**2,axis=1))/sigma_ph
                PH_q[i]       = np.argmax(Y_err_q>threshold_ph)/N_lyap
                if PH_q[i] == 0 and Y_err_q[0]<threshold_ph:
                    PH_q[i] = N_intt/N_lyap #(in case PH is larger than interval)


            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                if i<n_plot:
                    plt.subplot(n_plot,2,1+i*2)
                    xx = np.arange(U_wash[:,0].shape[0])/N_lyap

                    plt.plot(xx,U_wash[:U_wash.shape[0],i], 'k',label='True')
                    plt.plot(xx,Uh_wash[:U_wash.shape[0],i], '--r',label='ESN')
                    if quantum:
                        plt.plot(xx,Uh_wash_q[:U_wash.shape[0],i], '--b',label='RF-QRC')

                    plt.xlabel('Time[Lyapunov Times]')
                    plt.ylabel('$x$'+str(i+1))
                    if i==0:
                        plt.legend(ncol=2)
                        plt.title('Washout Phase')

                    plt.subplot(n_plot,2,2+i*2)
                    plt.axvline(PH[i])
                    xx = np.arange(Y_t[:,0].shape[0])/N_lyap
                    plt.plot(xx,Y_t[:,i], 'k')
                    plt.plot(xx,Yh_t[:,i], '--r')
                    if quantum:
                        plt.plot(xx,Yh_t_q[:,i], '--b',label='RF-QRC')
                    if i==0:
                        plt.legend(ncol=2)
                        plt.title('Testing Phase')

                    plt.xlabel('Time [Lyapunov Times]')

        # Percentiles of the prediction horizon
        PH_plot[j] = [np.quantile(PH,.75), np.median(PH), np.quantile(PH,.25)]
        print('PH quantiles [Lyapunov Times]:',
            PH_plot[j])
        print('')

        if quantum:
            # Percentiles of the prediction horizon
            PH_plot_q[j] = [np.quantile(PH_q,.75), np.median(PH_q), np.quantile(PH_q,.25)]
            print('PH quantiles [Lyapunov Times]:',
                PH_plot_q[j])
            print('')
        plt.show()

    return U_wash, Uh_wash, Y_t , Yh_t , PH_plot , Uh_wash_q, Yh_t_q , PH_plot_q


def lorenz63_timeseries_plot_quantum_only(N_test,N_tstart,N_intt,N_fwd,j,UU,U,U_test,N_washout,N_lyap,qu_ensemble,N_t,QESN,Woutt_qq,alpha_qq,plot=True,quantum=True):

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(UU)))
    threshold_ph = 0.5
    num_series = 0
    n_plot = 3

    PH_plot_q = np.zeros((qu_ensemble,3))

    print(U.shape, UU.shape, N_t)


    print('Realization    :',j+1)

    if quantum:
        Wout_q             = Woutt_qq[j]
        alpha              = alpha_qq[j]

        print('Quantum Hyperparameters:', QESN.rho_q, QESN.epsilon_q,QESN.sigma_in_q,QESN.tikh_q,'Seed:',j+1)

        # to store prediction horizon in the test set
        PH_q       = np.zeros(N_test)
        # to plot results

        if plot:
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout(h_pad=12)

        #run different test intervals
        for i in range(N_test):

            # data for washout and target in each interval
            U_wash    = U_test[num_series,N_tstart - N_washout +i*N_fwd : N_tstart + i*N_fwd].copy()
            Y_t       = U_test[num_series,N_tstart  +i*N_fwd           : N_tstart + i*N_fwd + N_intt].copy()


            if quantum:
                #washout for each interval
                Xa1_q   = QESN.quantum_openloop(U_wash, np.zeros(QESN.N_units),alpha) # here concatenation and bias out addition

                Uh_wash_q = np.dot(Xa1_q, Wout_q)

                # Prediction Horizon
                Yh_t_q        = QESN.quantum_closedloop(N_intt-1, Xa1_q[-1], Wout_q,alpha)[0]
                Y_err_q       = np.sqrt(np.mean((Y_t-Yh_t_q)**2,axis=1))/sigma_ph
                PH_q[i]       = np.argmax(Y_err_q>threshold_ph)/N_lyap
                if PH_q[i] == 0 and Y_err_q[0]<threshold_ph:
                    PH_q[i] = N_intt/N_lyap #(in case PH is larger than interval)


            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                if i<n_plot:
                    plt.subplot(n_plot,2,1+i*2)
                    xx = np.arange(U_wash[:,0].shape[0])/N_lyap

                    plt.plot(xx,U_wash[:U_wash.shape[0],i], 'k',label='True')
                    if quantum:
                        plt.plot(xx,Uh_wash_q[:U_wash.shape[0],i], '--b',label='RF-QRC')

                    plt.xlabel('Time[Lyapunov Times]')
                    plt.ylabel('$x$'+str(i+1))
                    if i==0:
                        plt.legend(ncol=2)
                        plt.title('Washout Phase')

                    plt.subplot(n_plot,2,2+i*2)
                    xx = np.arange(Y_t[:,0].shape[0])/N_lyap
                    plt.plot(xx,Y_t[:,i], 'k')
                    if quantum:
                        plt.plot(xx,Yh_t_q[:,i], '--b',label='RF-QRC')
                    if i==0:
                        plt.legend(ncol=2)
                        plt.title('Testing Phase')

                    plt.xlabel('Time [Lyapunov Times]')


        if quantum:
            # Percentiles of the prediction horizon
            PH_plot_q[j] = [np.quantile(PH_q,.75), np.median(PH_q), np.quantile(PH_q,.25)]
            print('PH quantiles [Lyapunov Times]:',
                PH_plot_q[j])
            print('')
        plt.show()

    return U_wash,  Y_t , Uh_wash_q, Yh_t_q , PH_plot_q
