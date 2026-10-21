import numpy as np
import matplotlib.pyplot as plt
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
mag_lin = np.abs(h)

# ploting section
fig, (log_plot, lin_plot) = plt.subplots(2, 1, figsize=(8, 10))
log_plot.plot(f, mag_dB, color='blue')
log_plot.set_xscale('log')
log_plot.set_title(f"{order} delta sigma mod (log)")
log_plot.set_xlabel("Normalized freq f/fs")
log_plot.set_ylabel("Amplitude [dB]")
log_plot.grid(True, which='both', linestyle=':')

lin_plot.plot(f, mag_lin, color='green')
lin_plot.set_title(f"{order} delta sigma mod (lin)")
lin_plot.set_xlabel("Normalized freq f/fs")
lin_plot.set_ylabel("Amplitude")
lin_plot.grid(True)

plt.savefig(f"{order}order_mod_NTF.png", bbox_inches='tight')
plt.tight_layout()
plt.show()