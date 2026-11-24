# *******************************************************************************
# This code realize computing and scaling DSM coefficients for given parameters
# in CIFF architecture and plot time response of scaled modulator
# *******************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import deltasigma

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
# cycles = int(np.round((f_in / f_s) * samples))
cycles = 61 # prime number to avoid idle tones
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

# print("Before scaling")
# print(xmax)

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

# print("After scaling")
# print(scaled_ABCD)
# print(xmax)

# ================================================
#             Time Domain DSM Response
# ================================================

samples_per_cycle = samples // cycles
end = 2 * samples_per_cycle  # Wycinek na 2 pełne okresy

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Górny wykres: Sygnał wejściowy vs Wyjście kwantyzatora
ax1.step(n[:end], v[:end], where='post', color='r', alpha=0.6, label="Quantizer Output (v)")
ax1.plot(n[:end], v_in[:end], color='k', linewidth=2, label="Input Signal (v_in)")
ax1.set_ylabel("Amplitude [Vref units]")
ax1.set_title("Time Domain DSM Response (Scaled CIFF 2nd Order)")
ax1.legend(loc="upper right")
ax1.grid(True, which='both', linestyle='--', alpha=0.7)

# Dolny wykres: Stany integratorów i granice xlim
ax2.plot(n[:end], xn[0, :end], color='b', alpha=0.8, label="1st Integrator State (X1)")
ax2.plot(n[:end], xn[1, :end], color='g', alpha=0.8, label="2nd Integrator State (X2)")
ax2.axhline(xlim, color='r', linestyle="--", label=f"OTA Limit (±{xlim:.2f})")
ax2.axhline(-xlim, color='r', linestyle="--")
ax2.set_xlabel("Samples [n]")
ax2.set_ylabel("State Level [Vref units]")
ax2.legend(loc="upper right")
ax2.grid(True, which='both', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("Time_Domain_Response_DSM.png", dpi=300, bbox_inches='tight')
plt.show()

# ================================================
#                NTF and STF Plot
# ================================================

ntf_scaled, stf_scaled = deltasigma.calculateTF(scaled_ABCD, k=1.0)

f_norm = np.logspace(-5, np.log10(0.5), 1000)
f_hz = f_norm * f_s
z = np.exp(2j * np.pi * f_norm)

ntf_scaled_z = deltasigma.evalTF(ntf_scaled, z)
ntf_scaled_z = 20 * np.log10(np.abs(ntf_scaled_z))

stf_scaled_z = deltasigma.evalTF(stf_scaled, z)
stf_scaled_z = 20 * np.log10(np.abs(stf_scaled_z))

stf_scaled_z = np.broadcast_to(stf_scaled_z, f_hz.shape)

plt.semilogx(f_hz, ntf_scaled_z, color='r', label="NTF")
plt.semilogx(f_hz, stf_scaled_z, color='b', label="STF")
plt.axvline(f_in_max, label="EEG_Band", linestyle="--")
plt.axvspan(f_hz[0], f_in_max, alpha=0.2)

plt.grid(True, which='both', linestyle='--', alpha=0.7)

plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.title(f"NTF and STF in Frequency Domain")
plt.legend()

plt.tight_layout()
plt.savefig("NTF_STF_scaled.png", dpi=300, bbox_inches='tight')
plt.show()



# =========================================
#                   FFT
# =========================================

hann_window = deltasigma.ds_hann(samples)

fft = np.fft.fft(v * hann_window) 
fft_norm = fft / (Vref * np.sum(hann_window) / 2) # Normalizing amplitude to Vref
fft_norm = fft_norm[0:samples // 2] # One Half on Spectrum taken into account
fft_db = 20 * np.log10(abs(fft_norm) + 1e-16)

freq_fft = np.linspace(0, f_s / 2, samples // 2, endpoint=False)

inband_bins = samples // (2 * OSR)

P_signal = (np.abs(fft_norm[cycles - 1]) ** 2) + (np.abs(fft_norm[cycles ]) ** 2) + (np.abs(fft_norm[cycles + 1]) ** 2) # side bins due to windowing
P_noise_A = np.abs(fft_norm[2:cycles - 1]) ** 2
P_noise_B = np.abs(fft_norm[cycles + 2:inband_bins + 1]) ** 2
P_noise = np.sum(P_noise_A) + np.sum(P_noise_B)

SQNR = 10 * np.log10(P_signal / P_noise)

SQNR_builtin = deltasigma.calculateSNR(fft_norm[:inband_bins], cycles) 

print(f"SQNR = {SQNR}")
print(f"SQNR_builtin = {SQNR_builtin}")

plt.semilogx(freq_fft[1:], fft_db[1:], color='b', label="FFT")
plt.axvline(f_in_max, label="EEG_Band", linestyle="--")
plt.axvspan(freq_fft[1], freq_fft[inband_bins], alpha=0.2)

plt.grid(True, which='both', linestyle='--', alpha=0.7)

plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dBFS]")
plt.title(f"Output Spectrum (SQNR = {SQNR:.2f} dB)")
plt.legend()

plt.tight_layout()
plt.savefig("DSM_FFT.png", dpi=300, bbox_inches='tight')
plt.show()