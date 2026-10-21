import numpy as np
import matplotlib.pyplot as plt
import scipy

bandwidth = 100 # Hz
OSR = 128 # Oversampling Ratio
fs = bandwidth * OSR # Sampling Frequency
samples = 2_000
total_ticks = samples * OSR # Number of Total Samples

fin = 1 # Input Signal Frequency

# =====================================================
#    Function of SC 1st Order Delta-Sigma Modulator
# =====================================================
def sc_delta_sigma(v_in, v_ref, a, b, total_ticks=samples*OSR):
    int_out = 0

    quantizer_out = np.zeros(total_ticks)

    comparator = 0

    for n in range(total_ticks):
        if(comparator == 1):
            int_out = int_out + a*v_in[n] - b*v_ref
        else:
            int_out = int_out + a*v_in[n] + b*v_ref

        comparator = int((int_out >= 0))
        quantizer_out[n] = comparator

    return quantizer_out

# =====================================================
#                   Input Signal
# =====================================================
t= np.arange(total_ticks) / fs # Discrete Time Sampling Vector (N * 1/fs = N * Ts)
amp1 = 0.01 # 10mV Amplitude of Input Signal
v_in1 = amp1 * np.sin(2 * np.pi * fin * t)

amp2 = 0.2 # 200mV
v_in2 = amp2 * np.sin(2 * np.pi * fin * t)

# =====================================================
#               Modulator parameters
# =====================================================
# Stable loop
C1_1 = 1
C2_1 = 1
Cf_1 = 1
v_ref1 = 0.1
a1 = C1_1/Cf_1
b1 = C2_1/(2*Cf_1)

# Clipping 
C1_2 = 1
C2_2 = 1
Cf_2 = 1
v_ref2 = 0.1
amp = 0.2
a2 = C1_2/Cf_2
b2 = C2_2/(2*Cf_2)

# =====================================================
#             Modulators output bitstream
# =====================================================
mod1_out = sc_delta_sigma(v_in1, v_ref1, a1, b1, total_ticks)
mod2_out = sc_delta_sigma(v_in2, v_ref2, a2, b2, total_ticks)

# Offset Cancelcation [1 0 1 0 0] -> [1 -1 1 -1 -1]
mod1_ac = (mod1_out * 2) - 1 
mod2_ac = (mod2_out * 2) - 1

# =====================================================
# Windowing output bitstream to prevent spectral leakage
# =====================================================
mod1_ac = mod1_ac * scipy.signal.windows.blackman(total_ticks, False)
mod2_ac = mod2_ac * scipy.signal.windows.blackman(total_ticks, False)

# =====================================================
#                         FFT
# =====================================================
fft1 = scipy.fft.fft(mod1_ac)
fft2 = scipy.fft.fft(mod2_ac)

freq = scipy.fft.fftfreq(total_ticks, 1/fs)

# =====================================================
#               Transform to dB scale
# =====================================================
freq = freq[:total_ticks//2]

fft1 = np.abs(fft1)
fft2 = np.abs(fft2)

fft1 = fft1[:total_ticks//2]
fft2 = fft2[:total_ticks//2]

db_fft1 = 20 * np.log10(fft1/np.max(fft1))
db_fft2 = 20 * np.log10(fft2/np.max(fft2))


# =====================================================
#                   Plot Spectrum
# =====================================================

plt.semilogx(freq, db_fft1, label="1st Order Delta Sigma Stable Loop")
# plt.semilogx(freq, db_fft2, label="1st Order Delta Sigma Unstable Loop")

plt.xlim([0, fs/2])
plt.ylim([np.min(db_fft1),5])

plt.grid(True, which="both", ls="--")

plt.legend()

plt.xlabel("Frequency [Hz]")
plt.ylabel("Signal Level [dB]")

plt.show()