# -------------------------------------------
# Authors: Tobias Ericsson, David Neinhardt |
# Date: 2025-02-27                          |
# Course Code: MTM021                       |
# -------------------------------------------
# Program som beräknar förskjutningar, stångkrafter och spänningar i en konstruktion

from utils import *
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np



# Skapar en figur med två horisontella subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# --- Given data ---
E_modul = 210.0e9  # [Pa]
sigma_s = 230.0e6  # [Pa]
L = 2.0  # [m]
A_0 = 78.5e-4  # [m^2]
A_Large = 2.0 * A_0
P = 150.0e3  # [N]

# 1.
# --- Numrera noder och frihetsgrader ---
# Nodernas koordinater
coords = np.array([
    [0, 0],         # Node 1
    [L, 0],         # Node 2
    [0, 2 * L],     # Node 3
    [L, 2 * L],     # Node 4
    [3 * L, 2 * L], # Node 5
    [L, 3 * L],     # Node 6
    [3 * L, 3 * L]  # Node 7
])
# Nodernas frihetsgrader
dofs = np.array([
    [1, 2],         # Node 1
    [3, 4],         # Node 2
    [5, 6],         # Node 3
    [7, 8],         # Node 4
    [9, 10],        # Node 5
    [11, 12],       # Node 6
    [13, 14]        # Node 7
])
# Elementens frihetsgrader
edof = np.array([
    [3, 4, 7, 8],       # Element 1   
    [3, 4, 5, 6],       # Element 2
    [1, 2, 7, 8],       # Element 3
    [1, 2, 5, 6],       # Element 4
    [5, 6, 7, 8],       # Element 5
    [5, 6, 11, 12],     # Element 6
    [7, 8, 11, 12],     # Element 7
    [7, 8, 9, 10],      # Element 8
    [9, 10, 11, 12],    # Element 9
    [11, 12, 13, 14],   # Element 10
    [9, 10, 13, 14],    # Element 11
])


# 2.
# --- Beräkna förskjutningar ---
ndofs = np.max(dofs)            # Antal frihetsgrader
K = np.zeros((ndofs, ndofs))    # Styvhetsmatris
f = np.zeros(ndofs)             # Kraftvektor

# Ansätter kraften P i nod 10
f[9] = -P 

# Areavektor för varje stång, där stång 4, 6 och 9 har dubbel area
A = np.array([A_0, A_0, A_0, A_Large, A_0, A_Large, A_0, A_0, A_Large, A_0, A_0])

# Beräknar en lokal styvhetsmatris för en stång
def bar_stiffness_matrix(E, A, ex, ey):
    L = np.sqrt((ex[1] - ex[0]) ** 2 + (ey[1] - ey[0]) ** 2)    # Hypotenusan för en stång
    C = (ex[1] - ex[0]) / L                                     # cosinus
    S = (ey[1] - ey[0]) / L                                     # sinus

    # Lokal styvhetsmatris
    k_local = (E * A / L) * np.array([
        [ C**2,  C*S, -C**2, -C*S],
        [ C*S,  S**2, -C*S, -S**2],
        [-C**2, -C*S,  C**2,  C*S],
        [-C*S, -S**2,  C*S,  S**2]
    ])
    return k_local

# Beräknar den globala styvhetsmatrisen
for i in range(len(edof)):
    ex, ey = coordxtr(edof[i].reshape(1, -1), coords, dofs) # Koordinater för stången
    ex, ey = ex.flatten(), ey.flatten()                     # Omvandlar till 1D-array

    Ke = bar_stiffness_matrix(E_modul, A[i], ex, ey)        # Lokala styvhetsmatrisen
    K = assem(edof[i], K, Ke)                               # Assemblerar den globala styvghetsmatrisen


bcdofs = np.array([1, 2, 3, 4])         # Noder med Dirichlet-villkor
bcvals = np.zeros(len(bcdofs))          # Dirichlet-villkor
a, Q = solveq(K, f, bcdofs, bcvals)     # Beräknar förskjutningar och krafter


# 3
# --- Beräkna stångkrafter och spänningar ---
Ed = extract_eldisp(edof, a)            # Förskjutningar för varje stång
Ex, Ey = coordxtr(edof, coords, dofs)   # Globala Koordinater för varje stång
sfac = 100     # Skalfaktor för att förstora förskjutningarna

# Första subplot: Förskjutna stänger
axes[0].set_title("Förskjutna stänger")
axes[0].set_xlabel("[m]")
axes[0].set_ylabel("[m]")
plt.sca(axes[0]) 
eldraw2(Ex, Ey, width=1, color="k")  # Ursprunglig struktur
eldisp2(Ex, Ey, Ed, sfac=sfac, width=1, color="r")  # Förskjutningar

nel =  len(edof)    # Antal element
N = np.zeros(nel)   # Stångkrafter

# Beräknar stångkrafterna
for el in range(nel):
    N[el] = bar2s(Ex[el], Ey[el], [E_modul, A[el]], Ed[el])

# Beräknar spänningarna
spänningar = N / A  

sorted_indices = np.argsort(spänningar)     # Sorterar stångarna efter spänning
N_sorted = N[sorted_indices]                # Stångkrafterna för de sorterade stängerna
spänn_sorted = spänningar[sorted_indices]   # Spänningarna för de sorterade stängerna

stång_med_min_spänn = sorted_indices[0]     # Stång med lägst spänning
stång_med_max_spänn = sorted_indices[-1]    # Stång med högst spänning


# 4
# --- Plottning av konstruktionen ---
# Ritar ut stängerna med färgkodning
for i in range(len(spänningar)):
    x_vals = [Ex[i, 0], Ex[i, 1]]  # x-koordinater för stången
    y_vals = [Ey[i, 0], Ey[i, 1]]  # y-koordinater för stången
    
    if spänningar[i] < 0:
        axes[1].plot(x_vals, y_vals, 'r', linewidth=1)  # Tryck: Rött
    elif spänningar[i] > 0:
        axes[1].plot(x_vals, y_vals, 'b', linewidth=1)  # Drag: Blått
    else:
        axes[1].plot(x_vals, y_vals, 'k', linewidth=1)  # Nollkraft: Svart

# Andra subplot: Stänger med tryck, drag och nollkraft
axes[1].set_title("Stänger med tryck (röd), drag (blå) och nollkraft (svart)")
axes[1].set_xlabel("[m]")
axes[1].set_ylabel("[m]")
axes[1].axis("equal")


# 5.
# --- Beräkna maximala tillåtna kraften och minimala tillåtna arean ---
P_Max = P / (np.max(np.abs(spänningar)) / sigma_s)  # Maximala tillåtna kraften
A_min = np.max(np.abs(N)) / sigma_s                 # Minsta tillåtna area



# ----------------------------------------------------------------------------------------------------------------------------------------
# --- Utskrift av resultat i tabell och plottning ---
table = [
    ["Knutförskjutningar", tabulate([[f"{val * 1000:.6f}" for val in a[i:i+2]] for i in range(0, len(a), 2)], tablefmt="grid"), "[mm]"],
    ["Stångkrafter", tabulate([[i+1, f"{N[i] * 1e-3:.6f}"] for i in range(len(N))], tablefmt="grid"), "[kN]"], 
    ["Spänningar", tabulate([[i+1, f"{spänningar[i] * 1e-6:.6f}"] for i in range(len(spänningar))], tablefmt="grid"), "[MPa]"], 
    ["Stång med lägst spänning", f"{stång_med_min_spänn + 1} ——— {spänningar[stång_med_min_spänn] * 1e-6:.6f}", "[MPa]"], 
    ["Stång med högst spänning", f"{stång_med_max_spänn + 1} ——— {spänningar[stång_med_max_spänn] * 1e-6:.6f}", "[MPa]"],
    ["Maximala tillåtna kraften", f"{(P_Max * 10**-3):.6f}", "[kN]"],
    ["Minimala tillåtna arean", f"{(A_min * 10**4):.6f}", "[cm²]"]
]

print("\nResultat:")
print(tabulate(table, headers=["Beskrivning", "Värde", "Enhet"], tablefmt="grid"))
plt.tight_layout()
plt.show()