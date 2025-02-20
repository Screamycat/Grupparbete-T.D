import numpy as np
import matplotlib.pyplot as plt
from utils import *

# Given data
E_modul = 210e9  # [Pa]
sigma_s = 230e6 # [Pa]
L = 2.5  # [m]
A_0 = 78.5e-1 # [m^2]
A_4_6_9 = 2 * A_0
P = 150e3  # [N]

# ## Topology
# Edof = np.array([
# [dofx1, dofy1, dofx2, dofy2],
# ...
# ], dtype=int )
# # Koordinater för varje nod:
# Coord = np.array([
# [0.0, 0.0],
# ...
# ])
# # x-koordinater för varje element
# Ex = np.array([
# [0.0, 0.0],
# ...
# ])
# # y-koordinater för varje element
# Ex = np.array([
# [0.0, 0.0],
# ...
# ])
# #Hjälpvariabler:
# nel = ... # Antal element
# ndofs = ... #Totalt antal frihetsgrader
# # Plot mesh (tips: använd eldraw2 i utils.py)
# ...
# # Fördefinera styvhetsmatrisen och kraftvektorn
# K = ...
# f = ...
# # Assemblera elemented
# for el in range(nel):
# #Räkna ut styvhetsmatrisen (Se föreläsningar eller Ekvation 11.18 i kursboken
# [Hållfasthetslära, Allmänna tillstånd])
# Ke = ...
# #Assemblera in element styvhetsmatrisen och globala matrisen
# K = assem(Edof[el, :], K, Ke)
# # Lägg till kraften P i lastvektorn:
# ...
# # Lös ekvations systemet (: använd solveq i utils.py)
# ...
# # Plotta deformerad mesh (: använd eldisp2 i utils.py)
# ...
# # Räkna ut krafter och spänningar i varje element
# for el in range(nel):
# #... tips: använd bar2s i utils.py