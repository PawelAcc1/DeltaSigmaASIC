import numpy as np
import matplotlib.pyplot as plt
import deltasigma

# Modulator Chosen Parameters
OSR = 256
order = 2
OBG = 2
Q_bits = 2
opt = 0
f0 = 0

amp_test = np.linspace(-100, 0, 50)

ntf = deltasigma.synthesizeNTF(
    order,
    OSR,
    opt,
    OBG,
    f0
)

snr_vector, amp_vector = deltasigma.simulateSNR(
    ntf,
    OSR,
    amp=amp_test,
    f0=f0,
    nlev=2**Q_bits
)

plt.plot(amp_vector, snr_vector, marker='o', linewidth=2, color='#1f77b4', label="SQNR")

plt.axhline(0, color='red', linestyle='--', label="0 dB Noise Floor")

plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel("Input Amplitude [dB]")
plt.ylabel("SQNR [dB]")
plt.title(f"SQNR vs Amplitude for OSR = {OSR}, ORDER = {order}, OBG = {OBG}, Q_bits = {Q_bits}")
plt.legend()

plt.tight_layout()
plt.savefig("Dynamic_Range_DSM.png", dpi=300)
plt.show()
