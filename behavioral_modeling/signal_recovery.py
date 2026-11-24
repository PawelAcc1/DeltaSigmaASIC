import numpy as np
import matplotlib.pyplot as plt
import deltasigma
import scipy.signal as sig

# ==================================================
#           DSM coefficients scaling
# ==================================================

# Modulator Parameters
order = 2
OSR = 256
opt = 0
OBG = 2
f0 = 0
Q_bits = 2
nlev = 2**Q_bits
Vref = nlev - 1 

# NTF Computation
ntf = deltasigma.synthesizeNTF(
    order,
    OSR,
    opt,
    OBG,
    f0
)

# Modulator Coefficients Computation
a, g, b, c = deltasigma.realizeNTF(ntf, 'CIFF')

# ABCD Matrix
unscaled_ABCD = deltasigma.stuffABCD(
    a,
    g,
    b,
    c,
    'CIFF'
)

# Input Sin Wave 
f_in = 50
f_in_max = 100 # Highest frequency of EEG signal
f_s = 2 * OSR * f_in_max
samples = 2**16
cycles = int(np.round((f_in / f_s) * samples))
f_in_coherent = (cycles / samples) * f_s

n = np.arange(samples)
amp = 0.8 * Vref
v_in = amp * np.sin(2 * np.pi * (f_in_coherent / f_s) * n)

# Simulating DSM for unscaled ABCD matrix
[v, xn, xmax, y] = deltasigma.simulateDSM(
    u=v_in,
    arg2=unscaled_ABCD,
    nlev=nlev,
    x0=0
)


# Scaling ABCD Matrix
xlim = 0.5 * Vref

# Scaling coefficients
alpha1 = xlim / xmax[0, 0]
alpha2 = xlim / xmax[1, 0]

scaled_ABCD = unscaled_ABCD.copy()

# 2. Input compensate
scaled_ABCD[0, :] = scaled_ABCD[0, :] * alpha1  # Integrator_1 input downscaling
scaled_ABCD[1, :] = scaled_ABCD[1, :] * alpha2  # Intergator_2 input downscaling

# 3. Output compensate
scaled_ABCD[:, 0] = scaled_ABCD[:, 0] / alpha1  # Integrator_1 output upscaling
scaled_ABCD[:, 1] = scaled_ABCD[:, 1] / alpha2  # Integrator_1 output upscaling



# Simulating DSM for scaled ABCD matrix
[v, xn, xmax, y] = deltasigma.simulateDSM(
    u=v_in,
    arg2=scaled_ABCD,
    nlev=nlev,
    x0=0
)