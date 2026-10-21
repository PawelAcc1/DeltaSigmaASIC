
import numpy as np
import deltasigma
import csv

# Modulator parameters
order = [1, 2, 3, 4]
OSR = [64, 128, 256, 512]
opt = 0
OBG = [1.2, 1.5, 2.0, 2.5, 3.0]
quantizer_bits = [1, 2, 3]
f0 = 0

# Amplitude vector in dB to simulate SNR
input_amp_vector = np.array([-18, -15, -12, -9, -6, -3, 0])

with open("SNR_comparison.csv", mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["OSR", "ORDER", "Q_BITS", "OBG", "SQNR_dB"])

    for osr in OSR:
        for ord_val in order:
            for q_bits in quantizer_bits:
                for obg in OBG:

                    ntf = deltasigma.synthesizeNTF(
                        ord_val, osr, opt, obg, f0
                    )

                    snr_vector, amp_vector = deltasigma.simulateSNR(
                        ntf, osr, nlev=2**q_bits, amp=input_amp_vector
                    )

                    SQNR = np.nanmax(snr_vector)

                    # Filtering obtained SQNR to reject unstable loop results
                    if(np.isnan(SQNR)):

                        SQNR = -20.0
                    writer.writerow([
                        osr, ord_val, q_bits, obg, round(SQNR, 2)
                    ])

                    print(
                        f"Saved SQNR:\n"
                        f"OSR = {osr}\n"
                        f"ORDER = {ord_val}\n"
                        f"QUANTIZER BITS = {q_bits}\n"
                        f"OBG = {obg}\n"
                        f"SQNR = {SQNR:.2f} dB"
                    )
