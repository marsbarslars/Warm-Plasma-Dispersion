import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from pathlib import Path

I = complex(0, 1)
C = 2.998e8  # m/s

# User-defined parameters
V_TH = {'ion': 0, 'electron': 0}  # m/s
EPSILON = {'ion': 1, 'electron': -1}
OMEGA_C = {'ion': 1e5, 'electron': 1.1e8}  # rad/s
OMEGA_P = {'ion': 7e4, 'electron': 3e6}  # rad/s
GAMMA = {'ion': 5/3, 'electron': 5/3}  # rad/s
THETA = np.pi / 2  # rad

NUM_POINTS = 200

k = np.logspace(-2, 5, NUM_POINTS)

OUTPUT_FILE_NAME = 'pictures/dispersion_relation'  # Default output file for the plot


def matrix(omega, k):
    '''
    Outputs the 3x3 matrix to take the determinant of.
    '''
    kx = k * np.sin(THETA)
    kz = k * np.cos(THETA)

    dielectric_tensor = np.zeros((3, 3), dtype=complex)
    for species in ['ion', 'electron']:
        A = np.zeros((3, 3), dtype=complex)
        A[0,0] = omega**2 - (GAMMA[species] * V_TH[species]**2 / 2) * kx**2
        A[1,0] = I * EPSILON[species] * omega * OMEGA_C[species]
        A[2,0] = -kx * kz * (GAMMA[species] * V_TH[species]**2 / 2)
        A[0,1] = -I * EPSILON[species] * omega * OMEGA_C[species]
        A[1,1] = omega**2
        A[2,1] = 0
        A[0,2] = -kx * kz * (GAMMA[species] * V_TH[species]**2 / 2)
        A[1,2] = 0
        A[2,2] = omega**2 - (GAMMA[species] * V_TH[species]**2 / 2) * kz**2
        dielectric_tensor += np.linalg.inv(A) * OMEGA_P[species]**2
    dielectric_tensor = np.eye(3) - dielectric_tensor

    n = C * k / omega
    nx = n * np.sin(THETA)
    nz = n * np.cos(THETA)

    n_matrix = np.zeros((3, 3), dtype=complex)
    n_matrix[0,0] = -nz**2
    n_matrix[1,0] = 0
    n_matrix[2,0] = nx * nz
    n_matrix[0,1] = 0
    n_matrix[1,1] = -n**2
    n_matrix[2,1] = 0
    n_matrix[0,2] = nx * nz
    n_matrix[1,2] = 0
    n_matrix[2,2] = -nx**2

    return n_matrix + dielectric_tensor

def determinant(omega, k):
    return np.linalg.det(matrix(omega, k))


omega_solution = np.zeros(NUM_POINTS)
for i in range(NUM_POINTS):
    omega_solution[i] = np.real(root_scalar(determinant, args=(k[i]), x0=1e6).root)

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(k*C/OMEGA_P['ion'], omega_solution/OMEGA_C['ion'])
ax.set_xlabel(r'$k / \omega_{c,i}$', fontsize=14)
ax.set_ylabel(r'$\omega c / \omega_{p,i}$', fontsize=14)
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()

save_file = input("Do you want to save the figure? (y/n): ")
if save_file.lower() == 'y':
    save_path = Path(__file__).resolve().parent / f"{OUTPUT_FILE_NAME}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100)
    print(f"Figure saved as {save_path}")


### Code below was used for testing.
# omegas = np.logspace(6.1, 8, 100)
# output = np.zeros(NUM_POINTS)
# for i in range(NUM_POINTS):
#     print(omegas[i])
#     output[i] = np.linalg.det(matrix(omegas[i], k))

# plt.plot(omegas, output)
# plt.show()
# omega_solution = np.real(root_scalar(determinant, args=(k), x0=1e6).root)
# print(omega_solution)