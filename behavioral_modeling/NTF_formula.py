import numpy as np
from scipy.signal import zpk2tf
import deltasigma

# modulator parameters for behavioral simulation
order = 3
OSR = 256
opt = 0
OBG = 3
f0 = 0

# NTF 
ntf = deltasigma.synthesizeNTF(order, OSR, opt, OBG, f0)

# frequency vector
f = np.arange(0.001, 0.5, 0.001)

# mapping frequency vector to Z domain
z = np.exp(1j * 2 * np.pi * f)

# frequency response of evaluated NTF to frequency vector
h = deltasigma.evalTF(ntf, z)

# dB scaling
mag_dB = 20 * np.log10(np.abs(h))

zeroes, poles, gain = ntf

# combining poles, zeroes and gain to obtain numerator and denominator of NTF formula
num, den = zpk2tf(zeroes, poles, gain)

num = np.real(num)
den = np.real(den)

print("~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"NTF zeroes: {zeroes}")
print("------------------------")
print(f"NTF poles: {poles}")
print("------------------------")
print(f"NTF gain: {gain}")
print("------------------------")
print(f"=== NTF polynomial ===")
print(f"Numerator's coefficients: {num}")
print(f"Denominator's coefficients: {den}")
print("~~~~~~~~~~~~~~~~~~~~~~~~")
