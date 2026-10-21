#=====================================================
# 1st order Discrete-Time Delta Sigma ADC
#=====================================================
import numpy as np
import matplotlib.pyplot as plt

# Sampling parameters
OSR = 128
samples = 1_000

# Voltage
v_in = 0.02
v_ref = 0.1

# Switched Integrator Capacitors Coefficients
# C1/Cf -> Input/Integrato
C1 = 1
Cf = 2
a = C1/Cf

# C2/Cf -> Vref/Integrator
C2 = 1
b = C2/(Cf*2)

int_out = 0

quantizer_out = np.zeros(samples*OSR)

comparator = 0

for n in range(OSR*samples):
    if(comparator == 1):
        int_out = int_out + a*v_in - b*v_ref
    else:
        int_out = int_out + a*v_in + b*v_ref

    comparator = int((int_out >= 0))
    quantizer_out[n] = comparator


y_mean = np.mean(quantizer_out)
v_reconstructed = (2*y_mean - 1)*(b/a)*v_ref

print("V_in = ", v_in)
print("V_reconstructed = ", v_reconstructed)
