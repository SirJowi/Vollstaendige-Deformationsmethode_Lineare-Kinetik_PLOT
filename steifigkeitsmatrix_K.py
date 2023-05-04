import numpy as np
import math

# Eingabeparameter

E = 2.1 * 10**8     # Elastizitätsmodul [kN/m^2]
I = 1.2 * 10**-4    # Flächenträgheitsmoment [m^4]
A = 1.0 * 10**-3    # Querschnittsfläche [m^2]
L12 = 5.5   # Stablänge [m]
alpha12 = 270   # Transformationswinkel [°]
L13 = 4
alpha13 = 0
L14 = 5
alpha14 = 36.87


# Transformationsmatrix
def T(alpha):
    rad = alpha * math.pi/180
    T = np.array([[np.cos(rad),    np.sin(rad),  0],
                  [- np.sin(rad),  np.cos(rad),  0],
                  [0,                0,              1]
                  ])
    return T


# Steifigkeitsmatrix

def K(E, I, A, L, lager, alpha):
    if lager == "e":
        K_lok = np.array([[ E*A/L,    0,              0           ],    # Steifigkeitsmatrix ki,k
                          [ 0,        12*E*I/L**3,    -6*E*I/L**2 ],
                          [ 0,        -6*E*I/L**2,    4*E*I/L     ]
                          ])
    K_glob = np.matmul(np.matmul(T(alpha), K_lok), np.transpose(T(alpha)))

    return np.round(K_lok)

"""
print(K(E, I, A, L12, "e", alpha12))

print(K(E, I, A, L13, "e", alpha13))

print(K(E, I, A, L14, "e", alpha14))

print(K(E, I, A, L12, "e", alpha12) + K(E, I, A, L13, "e", alpha13) + K(E, I, A, L14, "e", alpha14))
"""


v = np.array([-0.0005335, -0.0000975, -0.0003895])

print(np.array([0, -10, 13.75]) + np.matmul( K(E, I, A, L12, "e", alpha12), np.matmul(np.transpose(T(alpha12)), v)))

print(np.matmul( K(E, I, A, L13, "e", alpha13), np.matmul(np.transpose(T(alpha13)), v)))

print(np.array([0, -12.5, 10.417]) + np.matmul( K(E, I, A, L14, "e", alpha14), np.matmul(np.transpose(T(alpha14)), v)))

