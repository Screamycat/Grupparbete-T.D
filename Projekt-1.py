# -------------------------------------------
# Authors: Tobias Ericsson, David Neinhardt |
# Date: 2025-02-27                          |
# Course Code: MTM021                       |
# -------------------------------------------
# Ingenjörsuppgift GC-bro för MTM021

import numpy as np
import matplotlib.pyplot as plt



# --- Geometri [m] ---
h_f = 20e-3  # Flänstjocklek
b_f = 220e-3  # Flänsbredd
h_l = 520e-3  # Livhöjd
b_l = 10e-3  # Livbredd
L = 14.45  # Balkens längd
b = 4.095  # Brobredd
h = 125e-3  # Träskiva höjd
# --- Laster ---
w_s = 1.5e3  # Snölast [N/m^2]
w_t = 3.0e3  # Trängsellast [N/m^2]
P = 30e3  # Punktlast [N]
eta = 0.31
omega = 0.52
# --- Material ---
densitet_stål = 78e3  # [N/m^3]
densitet_trä = 6.6e3  # [N/m^3]
E_modul = 210e9  # Elasticitetsmodul [Pa]
sigma_y = 200e6  # Flytgräns [Pa]
alpha_stål = 1.2e-5  # Temperaturutvidgning [1/°C]
# --- Lastberäkningar ---
W_d = 2 * (2 * h_f * b_f + b_l * h_l) * densitet_stål + b * h * densitet_trä  # [N/m]
W_d = W_d / 2 # Beräkna per balk
W_s = w_s * b / 2  # [N/m]
W_t = w_t * b / 2  # [N/m]
W_tot = W_d + W_s + W_t  # Total linjelast [N/m]
P_b = P / 2  # Punktlast per balk


# GC4
# --- Reaktionskrafter ---
VB = P_b * eta + L/2 * (W_d + W_s) + W_t * omega * (L - omega * L / 2)
VA = P_b + L * (W_d + W_s + W_t * omega) - VB

V_f = VA + VB - P_b - W_d*L - W_s*L - W_t*L*omega
B_m = VA*L - P_b*L*(1-eta) - W_d*L**2/2 - W_s*L**2/2 - W_t*(L*omega)**2/2
print(f"\nVertikalkraft: {V_f} N")
print(f"Böjmoment: {B_m} Nm")

tol = 1e-10  # Tolerans
if V_f < tol and B_m < tol:
    print("\nBalken är i global jämvikt.")
else:
    print(f"\nBalken är INTE i global jämvikt.")

# --- Snittkrafter ---
x = np.linspace(0, L, 1000)  # Balkens längdindelning
def T(x):
    if 0 <= x < eta*L:
        return (W_d + W_s) * x - VA
    elif eta*L < x < L - omega*L:
        return P_b + (W_d + W_s) * x - VA
    elif L - omega*L < x <= L:
        return P_b + (W_d + W_s) * x + W_t*(x + omega*L - L) - VA
    
def M(x):
    if 0 <= x < eta*L:
        return (W_d + W_s) * x**2 / 2 - VA * x
    elif eta*L < x < L - omega*L:
        return P_b * (x - eta*L) + (W_d + W_s) * x**2 / 2 - VA * x
    elif L - omega*L < x <= L:
        return P_b * (x - eta*L) + (W_d + W_s) * x**2 / 2 + W_t * (x + omega*L - L)**2 / 2 - VA * x

T_x = np.array([T(xi) for xi in x])
M_x = np.array([M(xi) for xi in x])

# --- Maximala snittkrafter och böjmoment ---
# Hitta max snittkraft
T_max = np.max(T_x)
x_Tmax = x[np.argmax(T_x)]  # x-koordinat där max kraft uppstår

# Hitta max böjmoment
M_max = np.min(M_x) 
x_Mmax = x[np.argmin(M_x)]  # x-koordinat där max moment uppstår

# --- Plotta snittkraftsdiagram ---
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.plot(x, T_x *10**-3, label="Tvärkraft T(x)")
plt.plot(x_Tmax, T_max * 10**-3, 'ro', label=f"Maximal snittkraft: {T_max:.2f} N")
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("x [m]")
plt.ylabel("T [kN]")
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(x, M_x * 10**-3, label="Böjmoment M(x)", color='r')
plt.plot(x_Mmax, M_max * 10**-3, 'bo', label=f"Maximalt böjmoment: {M_max:.2f} Nm")
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("x [m]")
plt.ylabel("M [kNm]")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

print(f"\nMaximal snittkraft: {T_max:.2f} N vid x = {x_Tmax:.2f} m")
print(f"Maximalt böjmoment: {M_max:.2f} Nm vid x = {x_Mmax:.2f} m")


# GC5: 
# --- Yttröghetsmoment ---
A_f = b_f * h_f  # Flänsarea
d = (h_l / 2) + (h_f / 2)  # Avstånd från neutralaxeln till flänscentrum
I = 2 * ((b_f * h_f**3) / 12 + A_f * d**2) + (b_l * h_l**3) / 12
print(f"\nYttröghetsmoment I: {I:.6f} m^4")

# --- Normalspänningsberäkning ---
z_max = h_l/2 + h_f  # Överkanten av balken
z_min = -(h_l/2 + h_f)  # Underkanten av balken

sigma_max_drag = (M_max * z_min) / I
sigma_max_tryck = (M_max * z_max) / I

print(f"\nMaximal dragspänning: {sigma_max_drag / 1e6:.2f} MPa vid (x={x_Mmax:.2f} m, y=0, z={z_min:.3f} m)")
print(f"Maximal tryckspänning: {sigma_max_tryck / 1e6:.2f} MPa vid (x={x_Mmax:.2f} m, y=0, z={z_max:.3f} m)")

# --- Plotta normalspänningen ---
z_vals = np.linspace(z_min, z_max, 100)  # Höjd över tvärsnittet
sigma_vals = (M_max * z_vals) / I  # Normalspänning

plt.figure(figsize=(6, 8))
plt.plot(sigma_vals / 1e6, z_vals * 1e3, label="Normalspänning σ(z)")
plt.plot(sigma_max_drag / 1e6, z_min * 1e3, 'bo', label=f"Maximal dragspänning: {sigma_max_drag / 1e6:.2f} MPa")
plt.plot(sigma_max_tryck / 1e6, z_max * 1e3, 'bo', label=f"Maximal tryckspänning: {sigma_max_tryck / 1e6:.2f} MPa")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Spänning (MPa)")
plt.ylabel("Höjd över tvärsnitt (mm)")
plt.title("Normalspänning över balkens höjd")
plt.legend()
plt.grid()
plt.show()

# --- Jämförelse med flytgräns ---
if max(abs(sigma_max_drag), abs(sigma_max_tryck)) / 1e6 > sigma_y:
    print("\nBRO PLASTICERAR! Maxspänningen överskrider flytgränsen.")
else:
    print("\nBron klarar belastningen utan att plasticera.")