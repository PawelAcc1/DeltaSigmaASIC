# **************************************************************
# 3 Methods of computing SQNR for delta sigma ADC introduced in this script.
# 1) Via integration of quantization noise power in used band
# 2) Via formula provided in paper https://www.eecis.udel.edu/~vsaxena/courses/ece697A/s10/Lecture%20Notes/Lecture%2018%20Notes.pdf
# 3) Via built-in function from deltasigma library
# **************************************************************

import numpy as np
import deltasigma
import scipy.signal

# modulator parameters
order = 3
OSR = 256
opt = 0
OBG = 1.5
quantizer_bits = 1
f0 = 0

def SQNR_int (order=2, OSR=128, OBG=1.5, opt=0, quantizer_bits=1):

    f_max = 1 / (2 * OSR)
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
    P_e = 2 * quantization_error_total_power
    in_band_noise = np.trapz(P_e * mag_square, f_sig)

    # =================================================
    #       Max Range Signal Power Computation
    # =================================================
    max_signal_total_power = ((Vref / 2) ** 2) / 2

    # =================================================
    #               Real SQNR Computation
    # =================================================
    SQNR = 10 * np.log10(max_signal_total_power / in_band_noise)
    return SQNR

def SQNR_formula (order=2, OSR=128, quantizer_bits=1):

    modulator_product = ((20 * order + 10) * np.log10(OSR)) - (10 * np.log10((np.pi ** (2 * order))/((2 * order) + 1)))

    SQNR = (6.02 * quantizer_bits) + 1.76 + modulator_product 
    return SQNR

ntf = deltasigma.synthesizeNTF(order, OSR, opt, OBG, f0)
snr_vector, amp_vector = deltasigma.simulateSNR(ntf, OSR, nlev=2**quantizer_bits) # DO POPRAWY !!!
SQNR_builtin = np.max(snr_vector)

print("==================================")
print(f"Method_1: Integrating Noise Power \nSQNR = {SQNR_int(order, OSR, OBG, opt, quantizer_bits):.2f} dB")
print("----------------------------------")
print(f"Method_2: Formula Computation \nSQNR = {SQNR_formula(order, OSR, quantizer_bits):.2f} dB")
print("----------------------------------")
print(f"Method_3: Build_in Function Computation \nSQNR = {SQNR_builtin:.2f} dB")
print("==================================")