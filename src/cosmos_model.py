import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from data import *

# differnt E(z) models
def E_LCDM(z, Omega_m=Omega_m, Omega_Lambda=Omega_Lambda):
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)

def sinn(x, omega_k):
    return np.sinh(x) if omega_k > 0 else np.sin(x)

def mu_dynamic1(z, Omega_m=Omega_m, Omega_Lambda=Omega_Lambda):
    Omega_k = 1 - Omega_m - Omega_Lambda
    flag = Omega_k > 1e-10
    if Omega_k == 0: Omega_k = 1e-10
    integral, _ = quad(lambda z_prime: 1 / ((1 + z_prime)**2 * (1 + Omega_m * z_prime) - z_prime*(2 + z_prime) * Omega_Lambda)**(1/2), 0, z)
    dl = C0 / H0 * (1 + z) * sinn(np.sqrt(np.abs(Omega_k)) * integral, Omega_k) / np.sqrt(np.abs(Omega_k))
    return 5 * np.log10(dl) + 25

def E_dynamic2(z, Omega_m=Omega_m, w=-1):
    integral, _ = quad(lambda z_prime: (1 + w) / (1 + z_prime), 0, z)
    return np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m) * np.exp(3 * integral))

def E_dynamic3(z, w0=-1, wa=0):
    def integrand(z_prime):
        return (1 + w0 + wa * z_prime / (1 + z_prime)) / (1 + z_prime)
    integral, _ = quad(integrand, 0, z)
    return np.sqrt(Omega_m * (1 + z) ** 3 + (1 - Omega_m) * np.exp(3 * integral))

def E_dynamic4(z, w0=-1, wa=0):
    def integrand(z_prime):
        return (1 + w0 + wa / (1 + z_prime)) / (1 + z_prime)
    integral, _ = quad(integrand, 0, z)
    return np.sqrt(Omega_m * (1 + z) ** 3 + (1 - Omega_m) * np.exp(3 * integral))

# calculations
def mu(z, E_func, *args):
    integral, _ = quad(lambda z_prime: 1 / E_func(z_prime, *args), 0, z)
    integral = (C0 / H0) * (1 + z) * integral
    return 5 * np.log10(integral) + 25

if __name__ == "__main__":
    z_values = np.linspace(0.01, 2, 100)

    ### stimulation
    mu_LCDM   = [mu(z, E_LCDM) for z in z_values]
    mu_no_DE  = [mu(z, E_LCDM, 1, 0) for z in z_values]
    mu_all_DE = [mu(z, E_LCDM, 0, 1) for z in z_values]
    mu_dynamic_test = [mu_dynamic1(z) for z in z_values]

    # calculate the residue
    data["residue_LCDM"] = data["mua0"] - np.array([mu(z, E_LCDM) for z in data["z"]])

    ### plot the figure
    plt.figure(figsize=(10, 6))
    # data points
    plt.errorbar(data_gold["z"], data_gold["mua0"], yerr=data_gold["sigma_b"],
                fmt='*', color='green', label='Gold SN', ecolor='red', capsize=3)
    plt.errorbar(data_silver["z"], data_silver["mua0"], yerr=data_silver["sigma_b"],
                fmt='*', color='blue', label='Silver SN', ecolor='red', capsize=3)
    # models
    plt.plot(z_values, mu_LCDM, label=f"ΛCDM ({Omega_m}, {Omega_Lambda})", color="blue")
    # plt.plot(z_values, mu_dynamic_test, label="Dynamic DE ", color="black")
    plt.plot(z_values, mu_all_DE, label="Pure Dark Energy", color="green", linestyle="--")
    plt.plot(z_values, mu_no_DE, label="No Dark Energy", color="red", linestyle="--")
    # main plot settings
    plt.xlabel("Redshift (z)", fontsize=12)
    plt.ylabel("Distance Modulus $\mu$", fontsize=12)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.title("Comparison of Distance Modulus for Different Cosmologies(flat cosmos)", fontsize=14)
    # Inset plot for residuals
    ax_inset = plt.axes([0.5, 0.2, 0.35, 0.35])  # [left, bottom, width, height]
    ax_inset.scatter(data["z"], data["residue_LCDM"], color='blue', label='ΛCDM')
    ax_inset.axhline(0, color='black', linestyle='--')
    ax_inset.set_xlabel("z", fontsize=10)
    ax_inset.set_ylabel("Δμ", fontsize=10)
    ax_inset.set_title("Residuals in ΛCDM Model", fontsize=10)
    ax_inset.grid(True)
    plt.savefig('fig/cosmos_0.png')
    plt.show()