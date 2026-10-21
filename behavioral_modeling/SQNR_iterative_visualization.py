import numpy as np
import csv
import matplotlib.pyplot as plt

file_alias = 'SNR_comparison.csv'

# We want to enclose data of first row in 1D array
# to obtain order of modulator parameters 
parameters_order = np.genfromtxt(
    fname=file_alias,
    delimiter=',',
    max_rows=1,
    dtype=str
).tolist()


# Mapping columns of parameters to make extraction
# of parameters insensitive to its order in file
OSR_col = parameters_order.index('OSR')
ORDER_col = parameters_order.index('ORDER')
QBITS_col = parameters_order.index('Q_BITS')
OBG_col = parameters_order.index('OBG')
SQNR_col = parameters_order.index('SQNR_dB')


# 1. SQNR(OSR) for different 'order' parameter. Same: 'OBG', 'Q_BITS'
file_data = np.genfromtxt(
    fname=file_alias,
    dtype=float,
    delimiter=',',
    skip_header=1,
)

# Obtainig unique values of OSR and ORDER from file
OSR_all_val = file_data[:, OSR_col]
OSR_arr = np.unique(OSR_all_val)
OSR_arr = np.sort(OSR_arr)

ORDER_all_val = file_data[:, ORDER_col]
ORDER_arr = np.unique(ORDER_all_val)
ORDER_arr = np.sort(ORDER_arr)

OBG_all_val = file_data[:, OBG_col]
OBG_arr = np.unique(OBG_all_val)
OBG_arr = np.sort(OBG_arr)

QBITS_all_val = file_data[:, QBITS_col]
QBITS_arr = np.unique(QBITS_all_val)
QBITS_arr = np.sort(QBITS_arr)

Q_BITS_cosnt = 2.0
OBG_const = 2.0

# Creating logic masks for specified fields
logic_mask1 = (file_data[:, ORDER_col] == ORDER_arr[0]) & (file_data[:, QBITS_col] == Q_BITS_cosnt) & (file_data[:, OBG_col] == OBG_const)
logic_mask2 = (file_data[:, ORDER_col] == ORDER_arr[1]) & (file_data[:, QBITS_col] == Q_BITS_cosnt) & (file_data[:, OBG_col] == OBG_const)
logic_mask3 = (file_data[:, ORDER_col] == ORDER_arr[2]) & (file_data[:, QBITS_col] == Q_BITS_cosnt) & (file_data[:, OBG_col] == OBG_const)
logic_mask4 = (file_data[:, ORDER_col] == ORDER_arr[3]) & (file_data[:, QBITS_col] == Q_BITS_cosnt) & (file_data[:, OBG_col] == OBG_const)

# Extracting SQNR from data using prepared masks
SQNR_arr1 = file_data[logic_mask1]
SQNR_arr2 = file_data[logic_mask2]
SQNR_arr3 = file_data[logic_mask3]
SQNR_arr4 = file_data[logic_mask4]

# Ploting section
plt.plot(SQNR_arr1[:, OSR_col], SQNR_arr1[:, SQNR_col], label=f"Order = {int(ORDER_arr[0])}", marker='o')
plt.plot(SQNR_arr2[:, OSR_col], SQNR_arr2[:, SQNR_col], label=f"Order = {int(ORDER_arr[1])}", marker='o')
plt.plot(SQNR_arr3[:, OSR_col], SQNR_arr3[:, SQNR_col], label=f"Order = {int(ORDER_arr[2])}", marker='o')
plt.plot(SQNR_arr4[:, OSR_col], SQNR_arr4[:, SQNR_col], label=f"Order = {int(ORDER_arr[3])}", marker='o')

plt.grid(True, which='both', linestyle='--', alpha=0.7)

plt.xscale('log', base=2)

plt.xlabel("Oversampling Ratio (OSR)")
plt.ylabel("Signal to Quantization Noise Ratio (SQNR) [dB]")
plt.title(f"SQNR vs OSR for different modulator orders (OBG={OBG_const}, Q_bits={int(Q_BITS_cosnt)})")
plt.legend()
plt.tight_layout()

plt.savefig("SQNR_vs_OSR_2Qbits.png", dpi=300, bbox_inches='tight')

plt.show()

OSR_const = 256.0
Q_BITS_cosnt = 2.0

logic_mask1 = (file_data[:, ORDER_col] == ORDER_arr[0]) & (file_data[:, QBITS_col] == Q_BITS_cosnt) & (file_data[:, OSR_col] == OSR_const)
logic_mask2 = (file_data[:, ORDER_col] == ORDER_arr[1]) & (file_data[:, QBITS_col] == Q_BITS_cosnt) & (file_data[:, OSR_col] == OSR_const)
logic_mask3 = (file_data[:, ORDER_col] == ORDER_arr[2]) & (file_data[:, QBITS_col] == Q_BITS_cosnt) & (file_data[:, OSR_col] == OSR_const)
logic_mask4 = (file_data[:, ORDER_col] == ORDER_arr[3]) & (file_data[:, QBITS_col] == Q_BITS_cosnt) & (file_data[:, OSR_col] == OSR_const)

SQNR_arr1 = file_data[logic_mask1]
SQNR_arr2 = file_data[logic_mask2]
SQNR_arr3 = file_data[logic_mask3]
SQNR_arr4 = file_data[logic_mask4]

plt.plot(SQNR_arr1[:, OBG_col], SQNR_arr1[:, SQNR_col], label=f"Order = {int(ORDER_arr[0])}", marker='o')
plt.plot(SQNR_arr2[:, OBG_col], SQNR_arr2[:, SQNR_col], label=f"Order = {int(ORDER_arr[1])}", marker='o')
plt.plot(SQNR_arr3[:, OBG_col], SQNR_arr3[:, SQNR_col], label=f"Order = {int(ORDER_arr[2])}", marker='o')
plt.plot(SQNR_arr4[:, OBG_col], SQNR_arr4[:, SQNR_col], label=f"Order = {int(ORDER_arr[3])}", marker='o')

plt.grid(True, which='both', linestyle='--', alpha=0.7)

plt.xlabel("Out Band Gain")
plt.ylabel("Signal to Quantization Noise Ratio (SQNR) [dB]")
plt.title(f"SQNR vs OBG for different modulator orders (OSR={OSR_const}, Q_bits={int(Q_BITS_cosnt)})")
plt.legend()
plt.tight_layout()

plt.savefig("SQNR_vs_OBG_2Qbits.png", dpi=300, bbox_inches='tight')

plt.show()