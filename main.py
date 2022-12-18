"""Berechnung von Eigenfrequenzen -
Deformationsmethode - Lineare Kinetik, massebelegte Stäbe
18/12/2022
"""

import numpy as np

import matplotlib.pyplot as plt

# Determinante der Steifigkeitsmatrix K abhängig von omega berechnen

# Längsschwingung
def f_epsilon(omega, L, mu,  E, A, I):
    return L * np.sqrt((mu * omega ** 2) / (E * A))

# Querschwingung
def f_lambda(omega, L, mu,  E, A, I):
    return L * np.power((mu * omega**2) / (E * I), 1/4)


# Zelle 11 der Steifigkeitsmatrix
def k11(omega, L, mu,  E, A, I):

    K_11 = (E * A) / L * f_epsilon(omega, L, mu,  E, A, I) / np.tan(f_epsilon(omega, L, mu,  E, A, I))

    return K_11


# Zelle 22 der Steifigkeitsmatrix
def k22(omega, L, mu, E, A, I):

    # Phi-Beiwerte
    PHI_1 = (np.cosh(f_lambda(omega, L, mu,  E, A, I)) + np.cos(f_lambda(omega, L, mu,  E, A, I)) / 2)
    PHI_2 = (np.sinh(f_lambda(omega, L, mu,  E, A, I)) + np.sin(f_lambda(omega, L, mu,  E, A, I)) / 2)
    PHI_3 = (np.cosh(f_lambda(omega, L, mu,  E, A, I)) - np.cos(f_lambda(omega, L, mu,  E, A, I)) / 2)
    PHI_4 = (np.sinh(f_lambda(omega, L, mu,  E, A, I)) - np.sin(f_lambda(omega, L, mu,  E, A, I)) / 2)

    K_22 = (E * I * f_lambda(omega, L, mu,  E, A, I)**3) / L**3 *\
           (PHI_1**2 - PHI_2 * PHI_4) / (PHI_2 * PHI_3 - PHI_1 * PHI_4)

    return K_22


# Systemwerte ----------------------------------------------

E = 2.1 * 10 ** 8   # E-Modul
I = 3. * 10 ** -4    # Flächenträgheitsmoment
A = 5. * 10 ** -4    # Querschnitt
mu = 1.5    # Masseverteilung [t/m]

L14 = 9.     # Stablänge (14)
L12 = 5.    # Stablänge (12) = Stablänge (13)


# Wertebereich von x, in dem die Funktion ausgewertet werden soll
xlist = np.linspace(0, 150, num=1500)  # von 20 bis 115 und dazwischen 1500 Datenpunkte

# Berechnung der Determinante aus den einzelnen Steifigkeiten
k11_12 = k11(xlist, L12, mu, E, A, I)
k22_12 = k22(xlist, L12, mu, E, A, I)
k11_14 = k11(xlist, L14, mu, E, A, I)
k22_14 = k22(xlist, L14, mu, E, A, I)
# Determinante berechnen (Transformation nicht implementiert, vorher per Hand berechnet)
ylist = 0.9216 * k11_12**2 + 0.9216 * k22_12**2 + 2.1568 * k11_12 * k22_12 + 1.28 * k11_12 * k11_14 +\
        0.72 * k11_12 * k22_14 + 0.72 * k22_12 * k11_14 + 1.28 * k22_12 * k22_14 + k11_14 * k22_14

# Plot-Einstellungen
plt.ylim([-5 * 10**8, 5 * 10**8])   # y-Achsenbereich
plt.grid(True)  # Raster anzeigen lassen
plt.title("Auswertung der Eigenfrequenz")
plt.xlabel("Eigenkreisfrequenz $\omega$")
plt.ylabel("det K($\omega$)")

plt.plot(xlist, ylist)

plt.show()
