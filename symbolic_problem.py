import numpy as np
import sympy as sp

I = sp.I

k = sp.symbols('k', real=True, positive=True)
omega = sp.symbols('omega', real=True, positive=True)
v_th_e = sp.symbols('v_th_e', real=True, positive=True)
v_th_i = sp.symbols('v_th_i', real=True, positive=True)
omega_c_e = sp.symbols('omega_c_e', real=True, positive=True)
omega_c_i = sp.symbols('omega_c_i', real=True, positive=True)
omega_p_e = sp.symbols('omega_p_e', real=True, positive=True)
omega_p_i = sp.symbols('omega_p_i', real=True, positive=True)
c = sp.symbols('c', real=True, positive=True)
theta = sp.symbols('theta', real=True, positive=True)
gamma_e = sp.symbols('gamma_e', real=True, positive=True)
gamma_i = sp.symbols('gamma_i', real=True, positive=True)
V_TH = {'ion': v_th_i, 'electron': v_th_e}  # m/s
EPSILON = {'ion': 1, 'electron': -1}
OMEGA_C = {'ion': omega_c_i, 'electron': omega_c_e}  # rad/s
OMEGA_P = {'ion': omega_p_i, 'electron': omega_p_e}  # rad/s
GAMMA = {'ion': gamma_i, 'electron': gamma_e}  # rad/s
C = 2.998e8  # m/s
THETA = 0  # rad


kx = k * sp.sin(theta)
kz = k * sp.cos(theta)

dielectric_tensor = sp.zeros(3, 3)
for species in ['ion', 'electron']:
    A = sp.zeros(3, 3)
    A[0,0] = omega**2 - (GAMMA[species] * V_TH[species]**2 / 2) * kx**2
    A[1,0] = I * EPSILON[species] * omega * OMEGA_C[species]
    A[2,0] = -kx * kz * (GAMMA[species] * V_TH[species]**2 / 2)
    A[0,1] = -I * EPSILON[species] * omega * OMEGA_C[species]
    A[1,1] = omega**2
    A[2,1] = 0
    A[0,2] = -kx * kz * (GAMMA[species] * V_TH[species]**2 / 2)
    A[1,2] = 0
    A[2,2] = omega**2 - (GAMMA[species] * V_TH[species]**2 / 2) * kz**2
    dielectric_tensor += A.inv() * OMEGA_P[species]**2
dielectric_tensor = sp.eye(3) - dielectric_tensor

n = c * k / omega
nx = n * sp.sin(theta)
nz = n * sp.cos(theta)

n_matrix = sp.zeros(3, 3)
n_matrix[0,0] = -nz**2
n_matrix[1,0] = 0
n_matrix[2,0] = nx * nz
n_matrix[0,1] = 0
n_matrix[1,1] = -n**2
n_matrix[2,1] = 0
n_matrix[0,2] = nx * nz
n_matrix[1,2] = 0
n_matrix[2,2] = -nx**2

W = n_matrix + dielectric_tensor

solution = sp.solve(sp.det(W), omega)
print(solution)