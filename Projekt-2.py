import numpy as np
import matplotlib.pyplot as plt
from utils import *

# Given data
E_modul = 210e9  # [Pa]
sigma_s = 230e6 # [Pa]
L = 2  # [m]
A_0 = 78.5e-1 # [m^2]
A_Large = 2 * A_0
P = 150e3  # [N]


# 1.
# Numrera noder och frihetsgrader
# Noder:
coords = np.array([
    [0, 0],
    [L, 0],
    [0, 2*L],
    [L, 2*L],
    [3*L, 2*L],
    [L, 3*L],
    [3*L, 3*L]
])
#Frihetsgrader:
dofs = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10],
    [11, 12],
    [13, 14]
])
#Element
edof = np.array([
    [3, 4, 7, 8],       # Stång 1
    [3, 4, 5, 6],       # Stång 2
    [1, 2, 7, 8],       # Stång 3
    [1, 2, 5, 6],       # Stång 4
    [5, 6, 7, 8],       # Stång 5
    [5, 6, 11, 12],     # Stång 6
    [7, 8, 11, 12],     # Stång 7
    [7, 8, 9, 10],      # Stång 8
    [9, 10, 11, 12],    # Stång 9
    [11, 12, 13, 14],   # Stång 10
    [9, 10, 13, 14],    # Stång 11
])


# 2.
# Sätta upp styvhetsmatrisen och lösa ekvationssystemet
ndofs = np.max(edof)  # Antal frihetsgrader
K = np.zeros((ndofs, ndofs))  # Global styvhetsmatris
f = np.zeros(ndofs)  # Lastvektor
u = np.zeros(ndofs)  # Förskjutningsvektor

# P = 150 kN i nod 10
f[9] = -P 

# Stångareor
A = np.array([A_0, A_0, A_0, A_Large, A_0, A_Large, A_0, A_0, A_Large, A_0, A_0])

# Funktion för att skapa stångens styvhetsmatris
def bar_stiffness_matrix(E, A, ex, ey):
    L = np.sqrt((ex[1] - ex[0]) ** 2 + (ey[1] - ey[0]) ** 2)  # Längd
    C = (ex[1] - ex[0]) / L  # Cosinus
    S = (ey[1] - ey[0]) / L  # Sinus

    # Lokala styvhetsmatrisen
    k_local = (E * A / L) * np.array([
        [ C**2,  C*S, -C**2, -C*S],
        [ C*S,  S**2, -C*S, -S**2],
        [-C**2, -C*S,  C**2,  C*S],
        [-C*S, -S**2,  C*S,  S**2]
    ])
    return k_local

# Beräkna globala styvhetsmatrisen
for i in range(len(edof)):
    ex, ey = coordxtr(edof[i].reshape(1, -1), coords, dofs)  # Hämta nodkoordinater
    ex, ey = ex.flatten(), ey.flatten()

    Ke = bar_stiffness_matrix(E_modul, A[i], ex, ey)  # Beräkna stångens styvhetsmatris
    K = assem(edof[i], K, Ke)  # Montera in i globala styvhetsmatrisen

# Beräkna förskjutningar
bcdofs = np.array([1, 2, 3, 4])
bcvals = np.zeros(len(bcdofs))
a, Q = solveq(K, f, bcdofs, bcvals)
print("")
print("Uppgift 2:")
print("Knutförskjutningar: ", a, "[m]")

# 
Ed = extract_eldisp(edof, a)
Ex, Ey = coordxtr(edof, coords, dofs)

sfac = 1000  # Skalfaktor
plt.figure()
eldraw2(Ex, Ey, width=1, color="black")
eldisp2(Ex, Ey, Ed, sfac=sfac, width=1, color="r")
plt.show()


# 3.
nel = edof.shape[0]
N = np.zeros(nel)

for el in range(nel):
    N[el] = bar2s(Ex[el], Ey[el], [E_modul, A[el]], Ed[el])

sigma = N / A  # Elementspänningar

# Sortera efter sigma (minst -> störst)
sorted_indices = np.argsort(sigma)
N_sorted = N[sorted_indices]
sigma_sorted = sigma[sorted_indices]

# Identifiera stång med högst och lägst spänning
stång_med_min_sigma = sorted_indices[0]  # Index för stången med lägst spänning
stång_med_max_sigma = sorted_indices[-1] # Index för stången med högst spänning

print("\nUppgift 3:")
print("Stångkrafter:", N, "[Pa]")
print("Spänningar:", sigma, "[Pa]")
print("\nStång med lägst spänning: Stång", stång_med_min_sigma + 1, "med spänning", sigma_sorted[0], "[Pa]")
print("Stång med högst spänning: Stång", stång_med_max_sigma + 1, "med spänning", sigma_sorted[-1], "[Pa]")