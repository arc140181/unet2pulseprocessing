import numpy as np
import math as mat

## Pulse generation

def pulse_crrc(T,Ts,tau1,tau2):
    """
    Create a CR-RC pulse with height = 1.
    
    Parameters
    ----------
    T : float
        Duration of the pulse.
    Ts : float
        Sampling period.
    tau1, tau2 : float
        Time constants.

    Returns
    -------
    np.array
        The CR-RC pulse.
    
    """
    t = np.linspace(0,T*Ts,T)
    h = (tau1/((tau1 - tau2)))*(np.exp(-t/tau1) - np.exp(-t/tau2))    
    return h/np.max(h)


def pulse_crrcn(T,Ts,tau1,n):
    """
    Create a CR-(RC)^n pulse with height = 1.
    
    Parameters
    ----------
    T : float
        Duration of the pulse.
    Ts : float
        Sampling period.
    tau1: float
        Time constants for each stage.
    n: integer
        RC stages.

    Returns
    -------
    np.array
        The CR-(RC)^n pulse.
    
    """
    t = np.linspace(0,T*Ts,T)
    h = (1/mat.factorial(n)) * (t / tau1)**n * np.exp(-t / tau1)
    return h/np.max(h)

def pulse_sh(ka, ma, kb, mb, A, B, da, db):
    """
    Create a pulse with height = 1 with the shape explained in:
        V. T. Jordanov, “Real time digital pulse shaper with variable weighting function”.
        Nuclear Instruments and Methods in Physics Research A, 505 (2003) 347–351.
        
    Parameters
    ----------
    Explained in the referenced article.

    Returns
    -------
    np.array
        The obtained pulse.
        
    Examples
    --------
    pulse_sh(ka=50, ma=0, kb=50, mb=0, A=1, B=1, da=0, db=0)
        Triangular shaping.
    
    pulse_sh(ka=40, ma=0, kb=20, mb=60, A=1, B=1, da=10, db=0)
        Shaping for 1/f series noise.
    """

    ha = np.zeros((2*ka)+ma)
    hb = np.zeros((2*kb)+mb)

    for j in range(1,(2*ka)+ma):
        if (j<ka):
            ha[j-1] = ((j**2) + j) /2
        elif (j >= ka and j<= (ka + ma)):
            ha[j-1] = ka*(ka+1)/2
        elif (j > (ka + ma) and j < (2*ka + ma)):
            ha[j-1] = ((((2*ka)+ma-j)**2)+((2*ka)+ma-j))/2
    
    for j in range(1,(2*kb)+mb):
        if j < kb:
            hb[j-1] = ((((kb + 1)*2)*j)-(j**2)-j)/2
        elif j >= kb and j<= (kb + mb):
            hb[j-1] = kb*(kb+1)/2
        elif j > (kb + mb) and j < (2*kb + mb):
            hb[j-1] = (-((2*kb+mb-j)**2)+(2*kb+mb-j)*(2*kb+1))/2

    h1 = np.concatenate((np.zeros(da),A*ha,np.zeros(da))) 
    h2 = np.concatenate((np.zeros(db),B*hb,np.zeros(db)))
    h = h1 + h2
    
    return h/np.max(h)


## Differintegrals

def gdiff(y, x, gam):
    """
    Calculate the 'gam'-differintegral of y(x).
        
    Parameters
    ----------
    x : np.array
        The x-axis values.
    y : np.array
        The function value evaluated in x.
    gam : float
        The differintegral value.
        For positive integer values: dy^(gam)/dx^(gam).
        For negative integer values: integral(y dx^(-gam)).

    Returns
    -------
    np.array
        y(x) after applying the differintegral operator.

    """
    
    h = x[1] - x[0]
    
    lx = x.size
    dy = np.zeros(lx)
    w = np.ones(lx)
    
    for j in range(1,lx):
        w[j] = w[j-1]*(1-(gam+1)/j)
        
    for i in range(1,lx):
        dy[i] = np.sum(w[0:i+1] * y[i::-1]) / (h ** gam)
        
    return dy

# Noise color dependent from pulse shape. 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4...
def create_real_dataset_seg(pulse, length, smm, ssat, nmax, noisetype, freq, maxpileup=4):
    """
    Creates a time sequence of length 'length' along which pulses with shape 'pulse' turn up.
    The pulses are generated ramdomly in time and amplitude.
        
    Parameters
    ----------
    pulse : np.array
        Pulse shape (normalized to height=1).
    length : integer
        Length of the samples.
    smm : (float, float)
        Mininum and maximum pulse amplitude.
    ssat : float
        Saturation value, if it is below the pulse amplitude, pulses with higher height will appear cut off.
    nmax : float
        Noise amplitude
    noisetype : float
        Noise type at the output of the detector (e.g. 0 -> white, -0.5 -> 1/f noise, -1 -> brownian).
        Note that the pulse at the output of the detector is modelled as a step pulse and the noise
        is be filtered with the shape 'pulse'.
    freq : float
        Frequency of the appearance of pulses.
    maxpileup: integer
        Maximum of pile-up pulses. Ignore it if you will not use x_mask.
    

    Returns
    -------
    x_noise : np.array
        The time-sequence with noise.
        
    x : np.array
        The time-sequence without noise.
        
    x_mask : (np.array)^(maxpileup)
        Auxiliary value to control pulse pile-up. Each time that one pulse occurs, a counter c is
        increased. This counter is cyclic module 'maxpileup'. This counter select a window of events.
        The variable x_mask indicate when a pulse turn up in a concrete window.
        
    x_h : np.array
        The time-sequence with unfolded pulses.
        
    x_sep : (np.array)^(maxpileup)
        The same that x_mask but with pulses without noise.
    

    """    
    
    smin = smm[0]
    smax = smm[1]
    x = np.random.random_sample(length)
    x = (x > (1 - freq)) * ((smax - smin) * np.random.random_sample(length) + smin)
    x[x < (smax/1000)] = 0

    noise = nmax * gdiff(np.random.random_sample(length + 1) - 0.5, np.arange(0, length + 1, 1), noisetype)
    noise = np.diff(noise)
    
    x_noise = x + noise
        
    x_h = x

    x_sep_tmp = np.zeros((length,maxpileup))
    x_mask = np.zeros(length)
    c = 0
    for n in range(0, length):
        x_mask[n] = c
        if x[n] > 0:
            x_sep_tmp[n,c] = x[n]
            c = (c + 1) % maxpileup
            
    x = np.convolve(x, pulse)[0:length]
    x_noise = np.convolve(x_noise, pulse)[0:length]
    
    x[x > ssat] = ssat
    x_noise[x_noise > ssat] = ssat

    x_sep = x_sep_tmp
    
    return x_noise, x, x_mask, x_h, x_sep
