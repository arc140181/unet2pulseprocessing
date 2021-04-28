
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pulseutils import pulse_crrc, create_real_dataset_seg
from scipy import signal


PULSELEN = 128
pulse = pulse_crrc(T=PULSELEN, Ts=5e-8, tau1=1e-6, tau2=1e-7)


# Number of training pulses
NPULSESTRAIN = 800
# Number of test pulses
NPULSESTEST = 80
# Epochs of training
EPOCHS = 20
# Signal amplitude scaling
SMAX = 1.0
# Noise amplitude scaling
NMAX = 0.05
# Noise type ([0 -> white, -1 -> brownian])
NOISETYPE = 0.0

OCHANNELS = 3

LEN = 1024
FREQ = 0.01

x_train = np.zeros((NPULSESTRAIN, LEN, 1))
x_train_o2 = np.zeros((NPULSESTRAIN, LEN, 1))
for i in range(0,NPULSESTRAIN):
    x_train[i,:,0], _, _, x_train_o2_tmp, x_train_o3_tmp = create_real_dataset_seg(pulse, LEN, (0, SMAX), SMAX, NMAX, NOISETYPE, FREQ, maxpileup=OCHANNELS)
    x_train_o2[i,:,0] = np.roll(x_train_o2_tmp,1)
    
## NN1 (pulses to unfolding)
    
size = 3

inputs1 = tf.keras.layers.Input(shape=[None,  1], name="signal")
x1 = tf.keras.layers.Conv1D(32, size, strides=1, padding='same', activation=tf.nn.relu)(inputs1)
x1 = tf.keras.layers.Conv1D(32, size, padding='same', activation=tf.nn.relu)(x1)
x2 = tf.keras.layers.MaxPooling1D(2)(x1)
x2 = tf.keras.layers.Conv1D(64, size, padding='same', activation=tf.nn.relu)(x2)
x2 = tf.keras.layers.Conv1D(64, size, padding='same', activation=tf.nn.relu)(x2)
x3 = tf.keras.layers.MaxPooling1D(2)(x2)
x3 = tf.keras.layers.Conv1D(128, size, padding='same', activation=tf.nn.relu)(x3)
x3 = tf.keras.layers.Conv1D(128, size, padding='same', activation=tf.nn.relu)(x3)
x4 = tf.keras.layers.MaxPooling1D(2)(x3)
x4 = tf.keras.layers.Conv1D(256, size, padding='same', activation=tf.nn.relu)(x4)
x4 = tf.keras.layers.Conv1D(256, size, padding='same', activation=tf.nn.relu)(x4)
x5 = tf.keras.layers.MaxPooling1D(2)(x4)

x5 = tf.keras.layers.LSTM(512, return_sequences=True, activation=tf.nn.relu)(x5)
x5 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x5)
x5 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x5)
x5 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x5)

y4 = tf.keras.layers.UpSampling1D(2)(x5)
y4 = tf.keras.layers.Concatenate()([y4, x4])
y4 = tf.keras.layers.Conv1D(256, size, padding='same', activation=tf.nn.relu)(y4)
y4 = tf.keras.layers.Conv1D(256, size, padding='same', activation=tf.nn.relu)(y4)
y3 = tf.keras.layers.UpSampling1D(2)(y4)
y3 = tf.keras.layers.Dropout(0.2)(y3)
y3 = tf.keras.layers.Concatenate()([y3, x3])
y3 = tf.keras.layers.Conv1D(128, size, padding='same', activation=tf.nn.relu)(y3)
y3 = tf.keras.layers.Conv1D(128, size, padding='same', activation=tf.nn.relu)(y3)
y2 = tf.keras.layers.UpSampling1D(2)(y3)
y2 = tf.keras.layers.Dropout(0.2)(y2)
y2 = tf.keras.layers.Concatenate()([y2, x2])
y2 = tf.keras.layers.Conv1D(64, size, padding='same', activation=tf.nn.relu)(y2)
y2 = tf.keras.layers.Conv1D(64, size, padding='same', activation=tf.nn.relu)(y2)
y1 = tf.keras.layers.UpSampling1D(2)(y2)
y1 = tf.keras.layers.Dropout(0.2)(y1)
y1 = tf.keras.layers.Concatenate()([y1, x1])
y1 = tf.keras.layers.Conv1D(32, size, padding='same', activation=tf.nn.relu)(y1)
y1 = tf.keras.layers.Conv1D(32, size, padding='same', activation=tf.nn.relu)(y1)
x = tf.keras.layers.Conv1D(1, size, activation='linear', padding='same')(y1)


model1 = tf.keras.Model(inputs=inputs1, outputs=x)
model1.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model1.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

hist1 = model1.fit(x_train, x_train_o2,
    epochs=EPOCHS,
    batch_size=1,
    shuffle=True, 
    validation_split=0.2)


# Test the U-net

LEN = 2048
FREQ = 0.01 # 0.01 = 10^4 s^-1

x_test = np.zeros((NPULSESTEST, LEN, 1))
x_test_o2 = np.zeros((NPULSESTEST, LEN, 1))
for i in range(0,NPULSESTEST):
    x_test_tmp, _, _, x_test_o2_tmp, x_test_o3_tmp = create_real_dataset_seg(pulse, LEN, (0, SMAX), SMAX, NMAX, NOISETYPE, FREQ, maxpileup=OCHANNELS)
    x_test[i,:, 0] = x_test_tmp
    x_test_o2[i,:,0] = np.roll(x_test_o2_tmp,1)


y_pred1 = model1.predict(x_test)


## Plot results

# Learning

plt.figure()
plt.plot(hist1.epoch, hist1.history['loss'],'r',hist1.epoch, hist1.history['val_loss'],'r:')
plt.xlabel('Epoch')
plt.ylabel('Loss function')
plt.legend(['Train loss', 'Validation loss'])
plt.show()

# Graph

n = 0
l = np.arange(0, x_test.shape[1])

plt.figure()
plt.subplot(2,1,1)
plt.plot(l, x_test[n,:,0], color='black', linewidth=0.5, label='signal')
plt.grid(color='gray', linestyle=':', linewidth=1)
plt.ylabel('Normalized pulse amplitude')
plt.xlim(0, x_test.shape[1])
plt.title('Input signal')

plt.subplot(2,1,2)

THR = 0.03

l1 = l[x_test_o2[n,:,0]>THR]
l2 = l[y_pred1[n,:,0]>THR]

x_test_o2b = x_test_o2[n,x_test_o2[n,:,0]>THR,0]
y_pred1b = y_pred1[n,y_pred1[n,:,0]>THR,0]

markerline, stemlines, baseline = plt.stem(l1, x_test_o2b, linefmt='red', markerfmt='rx', label='Ideal')
plt.setp(stemlines, 'linewidth', 0.5)
plt.setp(markerline, 'linewidth', 0.5)
plt.setp(baseline, 'linewidth', 0.5)
plt.xlim(0, x_test.shape[1])
markerline, stemlines, baseline = plt.stem(l2, y_pred1b, linefmt='blue', markerfmt='b+', label='U-net')
plt.setp(stemlines, 'linewidth', 0.5)
plt.setp(markerline, 'linewidth', 0.5)
plt.setp(baseline, 'linewidth', 0.5)
plt.grid(color='gray', linestyle=':', linewidth=1)
plt.xlabel(r'time [$\mu$s]')
plt.ylabel('Normalized pulse amplitude')
plt.xlim(0, x_test.shape[1])
plt.title('Output signal')
plt.legend()
plt.show()

NPULSESTEST = 1
LEN = 240960
NMAX = 0.1
FREQ = 0.01

THR = 0.1
BINS = 80

x_test_hist = np.zeros((NPULSESTEST, LEN, 1))
x_test_hist_o2 = np.zeros((NPULSESTEST, LEN, 1))
for i in range(0,NPULSESTEST):
    x_test_hist_tmp, _, _, x_test_hist_o2_tmp, _ = create_real_dataset_seg(pulse, LEN, (SMAX/2, SMAX/2), SMAX, NMAX, NOISETYPE, FREQ, maxpileup=OCHANNELS)
    x_test_hist[i,:, 0] = x_test_hist_tmp
    x_test_hist_o2[i,:,0] = x_test_hist_o2_tmp
    
y_pred = model1.predict(x_test_hist)

x_test_hist_filt = np.convolve(x_test_hist_tmp, np.ones(5)/5)
x_test_hist_unf, _ = signal.deconvolve(x_test_hist_tmp,pulse[1:])

x_testb_hist = x_test_hist_tmp
x_test_hist_filtb = x_test_hist_filt
x_test_hist_unfb = x_test_hist_unf
y_pred1b = y_pred[0,:,0]


def FWHMHistogram(sig, thr, bins, smax, width=1):
    pospeaks, dicpeaks = signal.find_peaks(sig, height=thr, width=width)
    n1, bins1, _ = plt.hist(dicpeaks['peak_heights'], bins, range=(thr, (smax/2)*1.1), histtype='step', color='black')
    he = max(n1) * 1.05
    peaks, _ = signal.find_peaks(n1)
    n1[0:len(n1)//2] = 0
    pos = np.argmax(n1)
    results_half = signal.peak_widths(n1, [pos], rel_height=0.5)
    fwhm = results_half[0] * (bins1[pos] - bins1[pos-1])
    plt.text(bins1[pos-1]/1.6 - fwhm, np.max(n1)/2, 'FWHM=%.3f' % fwhm)
    plt.text(0.1, (np.max(n1)/4), '%d pulses' % dicpeaks['peak_heights'].shape[0])
    
    #plt.plot([bins1[pos]-(fwhm/2), bins1[pos]+(fwhm/2)], [max(n1)/2, max(n1)/2], color='blue', linewidth=2)

    return he


max1234 = np.zeros(5)

plt.figure()

plt.subplot(2,2,1)
max1234[1] = FWHMHistogram(x_testb_hist, THR, BINS, SMAX, width=10)
plt.title('Without filtering')
plt.ylabel('counts')

plt.subplot(2,2,2)
max1234[2] = FWHMHistogram(x_test_hist_filtb, THR, BINS, SMAX, width=10)
plt.title('With FIR filter')

plt.subplot(2,2,3)
max1234[3] = FWHMHistogram(x_test_hist_unfb, THR, BINS, SMAX)
plt.title('Linear unfolder')
plt.xlabel('channel')
plt.ylabel('counts')

plt.subplot(2,2,4)
max1234[4] = FWHMHistogram(y_pred1b, THR, BINS, SMAX)
plt.title('U-Net')
plt.xlabel('channel')

maxto = np.max(max1234)

for i in range(1,5):
    plt.subplot(2,2,i)
    plt.ylim(0,maxto)