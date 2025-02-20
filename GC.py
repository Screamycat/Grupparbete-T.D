import numpy as np
import matplotlib.pyplot as plt

# Given data
# Geometri [m]
h_f = 20e-3
b_f = 220e-3
h_l = 520e-3 
b_l = 10e-3
L = 14450e-3
b = 4095e-3
h = 125e-3
# Laster
w_s = 1.5e3  # [N/m^2]
w_t = 3.0e3  # [N/m^2]
P = 30e3  # [N]
eta = 0.31
omega = 0.52
# Material
densitet_stål = 78e3  # [N/m^3]
densitet_trä = 6.6e3  # [N/m^3]
E_modul = 210e9  # [Pa]
sigma_s = 200e6  # [Pa]
alpha_stål = 1.2e-5  # [1/°C]
# Lastberäkningar
W_d = (2 * h_f * b_f + b_l * h_l) * densitet_stål + b * h * densitet_trä  # [N/m]
W_s = w_s * b / 2  # [N/m]
W_t = w_t * b / 2  # [N/m]
W_tot = W_d + W_s + W_t  # Total utbredd last [N/m]
P_b = P / 2  # Punktlast per balk

# Beräkning av reaktionskrafter (jämvikt)
RA = P_b * (1.0-eta) + L/2*(W_d+W_s+W_t*omega)
RB = P_b*eta+L/2*(W_d+W_s) + w_t*(L-omega*L/2)

# Definiera x-axeln längs balken
x = np.linspace(0, L, 1000)

# Beräkna snittkrafter
T_x = RA - W_tot * x
M_x = RA * x - (W_tot * x**2) / 2

# Punktlastens bidrag (endast för M_x)
M_x[x >= L/2] -= P_b * (x[x >= L/2] - L/2)

# Hitta max böjmoment
M_max = np.max(M_x)
x_Mmax = x[np.argmax(M_x)]

# Plotta snittkraftsdiagram
plt.figure(figsize=(10,5))

plt.subplot(2,1,1)
plt.plot(x, T_x, label="Tvärkraft T(x)")
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("x [m]")
plt.ylabel("T [N]")
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(x, M_x, label="Böjmoment M(x)", color='r')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("x [m]")
plt.ylabel("M [Nm]")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

print(f"Maximalt böjmoment: {M_max:.2f} Nm vid x = {x_Mmax:.2f} m")
