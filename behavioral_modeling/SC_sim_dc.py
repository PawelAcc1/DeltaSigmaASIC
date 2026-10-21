import numpy as np
import matplotlib.pyplot as plt

def sc_delta_sigma(v_in, v_ref, a, b, OSR=128, samples=1_000):
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

    print("====================")
    print("V_in = ", v_in)
    print("V_ref = ", v_ref)
    print("a = ", a, " b = ", b)
    print("a * V_in = ", a*v_in, "\nb * V_ref = ", b*v_ref, "\nStable: ", bool((a*v_in <= b*v_ref)))
    print("V_reconstructed = ", v_reconstructed)
    print("====================")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# To make the modulator stable and prevent clipping we must ensure that the reference-integrator coupling (b * Vref)
# is stronger/higher than the input-integrator coupling (a * Vin). 
#                               (b * Vref) >= (a * Vin)
# Nevertheless too strong reference-integrator coupling increases noise since we try to compensate small input signal 
# using large steps
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Stable loop
C1_1 = 1
C2_1 = 1
Cf_1 = 1
v_ref1 = 0.1
v_in1 = 0.01
a1 = C1_1/Cf_1
b1 = C2_1/(2*Cf_1)

sc_delta_sigma(v_in1, v_ref1, a1, b1)

# Clipping 
C1_2 = 1
C2_2 = 1
Cf_2 = 1
v_ref2 = 0.1
v_in2 = 0.2
a2 = C1_2/Cf_2
b2 = C2_2/(2*Cf_2)

sc_delta_sigma(v_in2, v_ref2, a2, b2)