import numpy as np
import math as mat

## Pulse generation

def pulse_crrc(T,Ts,tau1,tau2):
    t = np.linspace(0,T*Ts,T)
    h = (tau1/((tau1 - tau2)))*(np.exp(-t/tau1) - np.exp(-t/tau2))    
    return h/np.max(h)


def pulse_crrcn(T,Ts,tau1,n):
    t = np.linspace(0,T*Ts,T)
    h = (1/mat.factorial(n)) * (t / tau1)**n * np.exp(-t / tau1)
    return h/np.max(h)

def pulse_sh(T, ka, ma, kb, mb, A, B, da, db):
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

def gdiff(y,x,gam):
    h = x[1] - x[0]
    
    lx = x.size
    dy = np.zeros(lx)
    w = np.ones(lx)
    
    for j in range(1,lx):
        w[j] = w[j-1]*(1-(gam+1)/j)
        
    for i in range(1,lx):
        dy[i] = np.sum(w[0:i+1] * y[i::-1]) / (h ** gam)
        
    return dy

## Error PHA and PSA

def errorH (x, xs):
    return (np.abs(np.max(xs) - np.max(x))) #/ np.max(xs)

def errorS (x, xs):
    lx = len(x)
    if lx != len(xs):
        return -1

    #return (np.sum((np.abs((xs - x) / xs)))) / lx
    return (np.sum((np.abs((xs - x))))) / lx

def errorHflat(x, xs):
    c1 = 0
    c2 = 0
    ha = []
    while c2 < len(xs):
        if xs[c2] == 0:
            if c2 == c1:
                c1 += 1
                c2 += 1
            else:
                ha.append(errorH(x[c1:c2],xs[c1:c2]))
                #ha.append(np.max(xs[c1:c2]))
                c2 += 1
                c1 = c2
        else:
            c2 += 1
    
    #print(np.array(ha))
    #return np.array(ha).mean()
    return np.array(ha)
    
        
def errorSflat_old(x, xs):    
    # Remove no-pulse values
    x = x[xs!=0]
    xs = xs[xs!=0]
    lx = len(x)
    if lx != len(xs):
        return -1
    
    #return (np.sum((np.abs((xs - x) / xs)))) / lx
    return (np.sum((np.abs(xs - x)))) / lx

def errorSflat(x, xs):   
    c1 = 0
    c2 = 0
    ha = []
    while c2 < len(xs):
        if xs[c2] == 0:
            if c2 == c1:
                c1 += 1
                c2 += 1
            else:
                ha.append(errorS(x[c1:c2],xs[c1:c2]))
                c2 += 1
                c1 = c2
        else:
            c2 += 1
    
    #print(np.array(ha))
    #return np.array(ha).mean()
    return np.array(ha)


def GetShiftingWindows(thelist, size):
    return np.array([thelist[x:x+size] for x in range(len(thelist) - size + 1)])

# Create dataset

def create_dataset(pulse, npulses, smax, nmax, noisetype, i2d=False, o2d=False):
    pulselen= len(pulse)
    if o2d == True:
        x_train = np.zeros((npulses, pulselen, 1, 1))
        for i in range(0, npulses):
            x_train[i,:,0,0] = pulse * smax * np.random.random_sample(1)
        if i2d == True:
            x_train_n = np.zeros((npulses, pulselen, 1, 1))
            for i in range(0, npulses):
                x_train_n[i,:,0,0] = x_train[i,:,0,0] + nmax * gdiff(np.random.random_sample(pulselen) - 0.5, np.arange(0, pulselen, 1), noisetype)
        else:
            x_train_n = np.zeros((npulses, pulselen))
            for i in range(0, npulses):
                x_train_n[i] = x_train[i,:,0,0] + nmax * gdiff(np.random.random_sample(pulselen) - 0.5, np.arange(0, pulselen, 1), noisetype)
    else:
        x_train = np.zeros((npulses, pulselen))
        for i in range(0, npulses):
            x_train[i] = pulse * smax * np.random.random_sample(1)
        if i2d == True:
            x_train_n = np.zeros((npulses, pulselen, 1, 1))
            for i in range(0, npulses):
                x_train_n[i,:,0,0] = x_train[i] + nmax * gdiff(np.random.random_sample(pulselen) - 0.5, np.arange(0, pulselen, 1), noisetype)
        else:
            x_train_n = np.zeros((npulses, pulselen))
            for i in range(0, npulses):
                x_train_n[i] = x_train[i] + nmax * gdiff(np.random.random_sample(pulselen) - 0.5, np.arange(0, pulselen, 1), noisetype)
    return x_train_n, x_train



def create_real_dataset_sol(pulse, samplelen, npulses, smax, nmax, noisetype, sol=1, i2d=False, o2d=False):    
    xh = smax * np.random.random_sample((npulses, sol))
    xs = np.random.randint(samplelen - 1, size=(npulses, sol))
    xs[:,0] = 0
    
    x = np.zeros((npulses, samplelen))
        
    for i in range(0, npulses):
        for j in range(0, sol):
            x[i, xs[i, j]] = xh[i, j]     
        noise = nmax * gdiff(np.random.random_sample(samplelen + 1) - 0.5, np.arange(0, samplelen + 1, 1), noisetype)
        noise = np.diff(noise)
        x_noise = x + noise
    
        x[i] = np.convolve(pulse, x[i])[0:samplelen]
        x_noise[i] = np.convolve(pulse, x_noise[i])[0:samplelen]
    
    if i2d == True:
        x_noise = x_noise.reshape((x_noise.shape[0], x_noise.shape[1], 1, 1))
    if o2d == True:
        x = x.reshape((len(x), 1, 1))

    return x_noise, x




# Noise color independent from pulse shape
def create_flat_dataset(pulse, npulses, smax, nmax, noisetype, sep, window=1, i2d=False, o2d=False):
    pulselen = len(pulse)

    x_train = np.zeros((npulses, pulselen))
    for i in range(0, npulses):
        x_train[i] = pulse * smax * np.random.random_sample(1)

    x_train = x_train.reshape((pulselen*npulses, 1, 1))

    for n in range(0,npulses):
        zeroarray = np.zeros(np.random.randint(0, sep))
        x_train = np.insert(x_train, ((npulses-n)*pulselen), zeroarray)    

    len_x_train = len(x_train)

    train_n = nmax * gdiff(np.random.random_sample(len_x_train+sep) - 0.5, np.arange(0, len_x_train+sep, 1), noisetype)
    train_n = np.diff(train_n)
    train_n = np.convolve(train_n, pulse)
    train_n = train_n[0:len_x_train]

    x_train_noise = x_train + train_n

    x_train_noise = GetShiftingWindows(x_train_noise, len_x_train - window + 1)
    x_train_noise = x_train_noise.transpose()
    x_train = x_train[window-1:]

    if i2d == True:
        x_train_noise = x_train_noise.reshape((x_train_noise.shape[0], x_train_noise.shape[1], 1, 1))

    if o2d == True:
        x_train = x_train.reshape((len(x_train), 1, 1))

    return x_train_noise, x_train

# Noise color dependent from pulse shape
def create_real_dataset(pulse, npulses, smax, nmax, noisetype, sep, window=1, i2d=False, o2d=False):
    x = np.zeros(npulses)
    for i in range(0, npulses):
        x[i] = smax * np.random.random_sample(1)

    for n in range(0, npulses):
        zeroarray = np.zeros(np.random.randint(0, sep))
        x = np.insert(x, (npulses - n), zeroarray)

    noise = nmax * gdiff(np.random.random_sample(len(x) + 1) - 0.5, np.arange(0, len(x) + 1, 1), noisetype)
    noise = np.diff(noise)
    
    x_noise = x + noise
    
    x = np.convolve(x, pulse)
    x_noise = np.convolve(x_noise, pulse)
    
    x_noise = GetShiftingWindows(x_noise, len(x) - window + 1)
    x_noise = x_noise.transpose()
    x = x[window - 1:]

    if i2d == True:
        x_noise = x_noise.reshape((x_noise.shape[0], x_noise.shape[1], 1, 1))
    if o2d == True:
        x = x.reshape((len(x),1,1))

    return x_noise, x

def create_dataset_pileup(pulse, npulses, smax, sep, window, i2d=False, o2d=False):
    x_train1, _ = create_flat_dataset(pulse, npulses, smax=smax, nmax=0, noisetype=0, sep=sep, window=window, i2d=i2d)
    x_train2, _ = create_flat_dataset(pulse, npulses, smax=smax, nmax=0, noisetype=0, sep=sep, window=window, i2d=i2d)
    
    len_x_train = np.min((x_train1.shape[0], x_train2.shape[0]))
    
    x_train = x_train1[0:len_x_train,:] + x_train2[0:len_x_train,:]
    
    if o2d==True:
        x_train_o = np.stack((x_train1[0:len_x_train,:], x_train2[0:len_x_train,:]), axis=3)
        x_train_o = x_train_o[...,0]
    else:
        if i2d==True:
            x_train_o = np.stack((x_train1[0:len_x_train,0,0,0], x_train2[0:len_x_train,0,0,0]), axis=1)
        else:
            x_train_o = np.stack((x_train1[0:len_x_train], x_train2[0:len_x_train]), axis=1)

    return x_train, x_train_o

def create_flat_dataset_seg(pulse, npulses, smax, nmax, noisetype, sep, window=1, i2d=False, o2d=False):
    pulselen = len(pulse)
    
    x_train = np.zeros((npulses, pulselen))
    x_train_mask = np.zeros((npulses, pulselen))
    x_train_h = np.zeros((npulses, pulselen))
    
    for i in range(0, npulses):
        h = smax * np.random.random_sample(1)
        x_train[i] = pulse * h
        x_train_mask[i] = (pulse > 0) * 1
        x_train_h[i] = x_train_mask[i] * h

    x_train = x_train.reshape((pulselen*npulses, 1, 1))
    x_train_mask = x_train_mask.reshape((pulselen*npulses, 1, 1))
    x_train_h = x_train_h.reshape((pulselen*npulses, 1, 1))

    for n in range(0,npulses):
        zeroarray = np.zeros(np.random.randint(0, sep))
        x_train = np.insert(x_train, ((npulses-n)*pulselen), zeroarray)
        x_train_mask = np.insert(x_train_mask, ((npulses-n)*pulselen), zeroarray)
        x_train_h = np.insert(x_train_h, ((npulses-n)*pulselen), zeroarray)

    len_x_train = len(x_train)
    
    train_n = nmax * gdiff(np.random.random_sample(len_x_train+sep) - 0.5, np.arange(0, len_x_train+sep, 1), noisetype)
    train_n = np.diff(train_n)
    train_n = np.convolve(train_n, pulse)
    train_n = train_n[0:len_x_train]

    x_train_noise = x_train + train_n

    x_train_noise = GetShiftingWindows(x_train_noise, len_x_train - window + 1)
    x_train_noise = x_train_noise.transpose()
    x_train = x_train[window-1:]
    x_train_mask = x_train_mask[window-1:]
    x_train_h = x_train_h[window-1:]

    if i2d == True:
        x_train_noise = x_train_noise.reshape((x_train_noise.shape[0], x_train_noise.shape[1], 1, 1))

    if o2d == True:
        x_train = x_train.reshape((len(x_train), 1, 1))
        x_train_mask = x_train_mask.reshape((len(x_train_mask), 1, 1))
        x_train_h = x_train_h.reshape((len(x_train_h), 1, 1))

    return x_train_noise, x_train, x_train_mask, x_train_h


# Noise color dependent from pulse shape. 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4...
def create_real_dataset_seg(pulse, length, smm, ssat, nmax, noisetype, freq, maxpileup=4):
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
    
    #x_sep = np.zeros((len(x),maxpileup))
    #for n in range(0, maxpileup):
    #    x_sep[:, n] = np.convolve(x_sep_tmp[:, n], pulse)
    #x_sep = x_sep[0:length, :]
    
    x_sep = x_sep_tmp
    
    return x_noise, x, x_mask, x_h, x_sep


# Noise color dependent from pulse shape. Counting the pulses 0 1 2 1 2 0 1 2 
def create_real_dataset_seg1(pulse, length, smm, nmax, noisetype, freq, maxpileup=4):
    smin = smm[0]
    smax = smm[1]
    x = np.random.random_sample(length)
    x = (x > (1 - freq)) * ((smax - smin) * np.random.random_sample(length) + smin)

    noise = nmax * gdiff(np.random.random_sample(len(x) + 1) - 0.5, np.arange(0, len(x) + 1, 1), noisetype)
    noise = np.diff(noise)
    
    x_noise = x + noise
    
    x_norm = (x > 0) * 1
    x_mask = np.convolve(x_norm, np.ones(len(pulse)))
    
    x_mask_end = np.zeros(len(x_mask))
    cell = np.zeros(maxpileup)
    a = 0
    for c in range(0, len(x_norm)):
        if x_norm[c]==1:
            a = 0
            flag = False
            while a < maxpileup and flag==False:
                if cell[a]==0:
                    cell[a] = len(pulse)
                    flag = True
                else:
                    a += 1
        
        if not all(cell==0):
            x_mask_end[c] = a + 1
            cell[cell>0] -= 1

    x_sep_tmp = np.zeros((len(x),maxpileup))
    for n in range(1, maxpileup):
        x_sep_tmp[:, n-1] = (x_mask_end[0:len(x)] == n) * x
        
    x_h = x
    x = np.convolve(x, pulse)[0:length]
    x_noise = np.convolve(x_noise, pulse)[0:length]
    
    
    #x_sep = x_sep_tmp
    x_sep = np.zeros((len(x),maxpileup))
    for n in range(0, maxpileup):
        x_sep[:, n] = np.convolve(x_sep_tmp[:, n], pulse)[0:length]
    x_sep = x_sep[0:length, :]
    
    # Detector saturation
    x_noise[x_noise > smax] = smax
    
    return x_noise, x, x_mask_end, x_h, x_sep


# Noise color dependent from pulse shape. Segmentation prop. to pulse height. Counting the pulses 0 1 2 1 2 0 1 2 
def create_real_dataset_seg2(pulse, length, smax, nmax, noisetype, freq, maxpileup=4):
    x = np.random.random_sample(length)
    x = (x > (1 - freq)) * smax * np.random.random_sample(length)
    
    noise = nmax * gdiff(np.random.random_sample(len(x) + 1) - 0.5, np.arange(0, len(x) + 1, 1), noisetype)
    noise = np.diff(noise)
    
    x_noise = x + noise
    
    x_norm = (x > 0) * 1
    x_mask = np.convolve(x_norm, np.ones(len(pulse)))
    
    x_mask_end = np.zeros(len(x_mask))
    cell = np.zeros(maxpileup)
    a = 0
    for c in range(0, len(x_norm)):
        if x_norm[c]==1:
            a = 0
            flag = False
            while a < maxpileup and flag==False:
                if cell[a]==0:
                    cell[a] = len(pulse)
                    flag = True
                else:
                    a += 1
        
        if not all(cell==0):
            x_mask_end[c] = a + 1    
            cell[cell>0] -= 1

    x_h = x

    x_sep_tmp = np.zeros((len(x),maxpileup))
    for n in range(1, maxpileup):
        x_sep_tmp[:, n-1] = (x_mask_end[0:len(x)] == n) * x
        
    
    x = np.convolve(x, pulse)[0:length]
    x_noise = np.convolve(x_noise, pulse)[0:length]
    
    x_sep = np.zeros((len(x),maxpileup))
    for n in range(0, maxpileup):
        x_sep[:, n] = np.convolve(x_sep_tmp[:, n], pulse)
    x_sep = x_sep[0:length, :]
    
    return x_noise, x, x_mask_end, x_h, x_sep


# Noise color dependent from pulse shape. Counting the pulses 0 1 2 3 0 1 2 0 ...
def create_real_dataset_seg4(pulse, length, smax, nmax, noisetype, freq, maxpileup=4):
    x = np.random.random_sample(length)
    x = (x > (1 - freq)) * smax * np.random.random_sample(length)

    noise = nmax * gdiff(np.random.random_sample(len(x) + 1) - 0.5, np.arange(0, len(x) + 1, 1), noisetype)
    noise = np.diff(noise)
    
    x_noise = x + noise
    
    x_norm = (x > 0) * 1
    x_mask = np.convolve(x_norm, np.ones(len(pulse)))[0:length]
    
    x_mask_end = np.zeros(length)
    c = 0
    while c < len(x_mask):
        if x_mask[c] > 0:
            d = c
            while d < length and x_mask[d] > 0:
                d += 1
            x_mask_seg = x_mask[(c-1):(d-1)]
            d_x_mask_seg = np.diff(x_mask_seg)
            d_x_mask_seg_abs = (d_x_mask_seg > 0) * d_x_mask_seg
            if len(np.cumsum(d_x_mask_seg_abs)) > 0:
                x_mask_end[c:(d-1)] = np.cumsum(d_x_mask_seg_abs)
            c = d
        c += 1
    
    x_h = x

    x_sep_tmp = np.zeros((len(x),maxpileup))
    for n in range(1, maxpileup):
        x_sep_tmp[:, n-1] = (x_mask_end[0:len(x)] == n) * x
            
    x = np.convolve(x, pulse)[0:length]
    x_noise = np.convolve(x_noise, pulse)[0:length]

    x[x > smax] = smax
    x_noise[x_noise > smax] = smax
    
    x_sep = np.zeros((len(x),maxpileup))
    for n in range(0, maxpileup):
        x_sep[:, n] = np.convolve(x_sep_tmp[:, n], pulse)[0:length]
        #x_sep[:, n] = x_sep_tmp[:, n]
    
    return x_noise, x, x_mask_end, x_h, x_sep
