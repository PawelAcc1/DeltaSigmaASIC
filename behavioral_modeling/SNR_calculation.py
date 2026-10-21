import numpy as np
import deltasigma
import scipy.signal

# =================================================
#               Modulator Parameters 
# =================================================
order = 3
OSR = 256
opt = 0
OBG = 3
f0 = 0
f_max = 1 / (2 * OSR)
quantizer_bits = 3
Vref = 1

# =================================================
#                  NTF calculation
# =================================================
ntf = deltasigma.synthesizeNTF(order, OSR, opt, OBG, f0)

f_sig = np.linspace(0, f_max, 1000)

z = np.exp(1j * 2 * np.pi * f_sig)

h = deltasigma.evalTF(ntf, z)

mag_square = np.abs(h)**2 

# =================================================
#       Quantization Noise Power Computation
# =================================================
lsb = Vref / 2**(quantizer_bits)
quantization_error_total_power = (lsb ** 2)/12 
P_e = 2 * quantization_error_total_power # Both halves of Gaussian Distribution of Quantization Noise taken into account
in_band_noise = np.trapz(P_e * mag_square, f_sig) # Noise Power Integration 

# =================================================
#       Max Range Signal Power Computation
# =================================================
max_signal_total_power = ((Vref / 2) ** 2) / 2

# =================================================
#               Real SQNR Computation
# =================================================
SQNR = 10 * np.log10(max_signal_total_power / in_band_noise)

print("==================================")
print(f"Modulator Order =  {order} \nOSR = {OSR}\nOBG = {OBG}\nQuantizer Bits = {quantizer_bits}")
print(f"SQNR = {SQNR}")
print("==================================")