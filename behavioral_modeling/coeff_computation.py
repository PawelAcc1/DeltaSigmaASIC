import numpy as np
import deltasigma

order = 2
OSR = 256
opt = 0
OBG = 2
f0 = 0

ntf = deltasigma.synthesizeNTF(
    order,
    OSR,
    opt,
    OBG,
    f0
)

a, g, b, c = deltasigma.realizeNTF(ntf, 'CIFF')

print(f"a coeffs: {a}")
print(f"g coeffs: {g}")
print(f"b coeffs: {b}")
print(f"c coeffs: {c}")