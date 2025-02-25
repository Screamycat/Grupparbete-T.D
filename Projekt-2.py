import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math
from utils import *

# Given data
E_modul = 210e9  # [Pa]
sigma_s = 230e6  # [Pa]
L = 2  # [m]
A_0 = 78.5e-3  # [m^2]
A_Large = 2 * A_0
P = 150e3  # [N]

# 1.
# Numrera noder och frihetsgrader
coords = np.array([
    [0, 0],
    [L, 0],
    [0, 2 * L],
    [L, 2 * L],
    [3 * L, 2 * L],
    [L, 3 * L],
    [3 * L, 3 * L]
])
dofs = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10],
    [11, 12],
    [13, 14]
])
edof = np.array([
    [3, 4, 7, 8],       
    [3, 4, 5, 6],       
    [1, 2, 7, 8],       
    [1, 2, 5, 6],       
    [5, 6, 7, 8],       
    [5, 6, 11, 12],     
    [7, 8, 11, 12],     
    [7, 8, 9, 10],      
    [9, 10, 11, 12],    
    [11, 12, 13, 14],   
    [9, 10, 13, 14],    
])

# 2.
ndofs = np.max(edof)  
K = np.zeros((ndofs, ndofs))  
f = np.zeros(ndofs)  
u = np.zeros(ndofs)  

f[9] = -P 

A = np.array([A_0, A_0, A_0, A_Large, A_0, A_Large, A_0, A_0, A_Large, A_0, A_0])

def bar_stiffness_matrix(E, A, ex, ey):
    L = np.sqrt((ex[1] - ex[0]) ** 2 + (ey[1] - ey[0]) ** 2)  
    C = (ex[1] - ex[0]) / L  
    S = (ey[1] - ey[0]) / L  

    k_local = (E * A / L) * np.array([
        [ C**2,  C*S, -C**2, -C*S],
        [ C*S,  S**2, -C*S, -S**2],
        [-C**2, -C*S,  C**2,  C*S],
        [-C*S, -S**2,  C*S,  S**2]
    ])
    return k_local

for i in range(len(edof)):
    ex, ey = coordxtr(edof[i].reshape(1, -1), coords, dofs)  
    ex, ey = ex.flatten(), ey.flatten()

    Ke = bar_stiffness_matrix(E_modul, A[i], ex, ey)  
    K = assem(edof[i], K, Ke)  

bcdofs = np.array([1, 2, 3, 4])
bcvals = np.zeros(len(bcdofs))
a, Q = solveq(K, f, bcdofs, bcvals)
print("")
print("Uppgift 2:")
print("Knutförskjutningar: ", a, "[m]")


# 3
# Beräkna stångkrafter och spänningar
Ed = extract_eldisp(edof, a)
Ex, Ey = coordxtr(edof, coords, dofs)

sfac = 1000
plt.figure()
eldraw2(Ex, Ey, width=1, color="black")
eldisp2(Ex, Ey, Ed, sfac=sfac, width=1, color="r")
plt.show()

nel = edof.shape[0]
N = np.zeros(nel)

for el in range(nel):
    N[el] = bar2s(Ex[el], Ey[el], [E_modul, A[el]], Ed[el])

sigma = N / A  

sorted_indices = np.argsort(sigma)
N_sorted = N[sorted_indices]
sigma_sorted = sigma[sorted_indices]

stång_med_min_sigma = sorted_indices[0]  
stång_med_max_sigma = sorted_indices[-1]  

print("\nUppgift 3:")
print("Stångkrafter:", N, "[N]")
print("Spänningar:", sigma, "[Pa]")
print("\nStång med lägst spänning: Stång", stång_med_min_sigma + 1, "med spänning", sigma_sorted[0], "[Pa]")
print("Stång med högst spänning: Stång", stång_med_max_sigma + 1, "med spänning", sigma_sorted[-1], "[Pa]")

# 4.
# Visualisering
plt.figure()

for i in range(len(sigma)):
    x_vals = [Ex[i, 0], Ex[i, 1]]
    y_vals = [Ey[i, 0], Ey[i, 1]]

    if sigma[i] < 0:
        plt.plot(x_vals, y_vals, 'r', linewidth=2)  # Tryck: Rött
    elif sigma[i] > 0:
        plt.plot(x_vals, y_vals, 'b', linewidth=2)  # Drag: Blått
    else:
        plt.plot(x_vals, y_vals, 'k', linewidth=2)  # Nollkraft: Svart

plt.xlabel("X-koordinater")
plt.ylabel("Y-koordinater")
plt.title("Stänger med tryck (röd), drag (blå) och nollkraft (svart)")
plt.axis("equal")
plt.show()


# 5.
# Visualisering
P_max = math.inf
for i in range(len(A)):
    local_p = sigma_s * A[i]
    if local_p < P_max:
        P_max = local_p

A_min = -math.inf
for i in range(len(A)):
    local_a = P / sigma_s
    if local_a > A_min:
        A_min = local_a

print("\nUppgift 5:")
print("Maximala tillåtna kraften: ", P_max * 10**-3, "[kN]")
print("Minimala tvärsnittsarean: ", A_min, "[m^2]")