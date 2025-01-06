import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Constants and parameters
C0 = 3e5  # light speed, km/s
H0 = 70  # Hubble constant, km/s/Mpc
Omega_m = 0.27  # matter density parameter
Omega_Lambda = 0.73  # dark energy density parameter

# E(z) functions
def E_LCDM(z, Omega_m=Omega_m, Omega_Lambda=Omega_Lambda):
    return np.sqrt(Omega_m * (1 + z) ** 3 + Omega_Lambda)

# Luminosity distance
def luminosity_distance(z, E_func, *args):
    integral, _ = quad(lambda z_prime: 1 / E_func(z_prime, *args), 0, z)
    return (C0 / H0) * (1 + z) * integral

def mu(z, E_func, *args):
    return 5 * np.log10(luminosity_distance(z, E_func, *args)) + 25

# Binning function for data
def bin_data(z_data, mu_data, sigma_data, n_bins):
    """
    Bin the data into `n_bins` bins based on redshift (z).
    Returns the binned z, mu, and sigma values.
    """
    bins = np.linspace(min(z_data), max(z_data), n_bins + 1)
    z_binned = []
    mu_binned = []
    sigma_binned = []
    
    for i in range(len(bins) - 1):
        mask = (z_data >= bins[i]) & (z_data < bins[i + 1])
        if np.sum(mask) > 0:
            z_mean = np.mean(z_data[mask])
            mu_mean = np.mean(mu_data[mask])
            sigma_mean = np.sqrt(np.sum(sigma_data[mask] ** 2) / np.sum(mask))  # Combined error
            z_binned.append(z_mean)
            mu_binned.append(mu_mean)
            sigma_binned.append(sigma_mean)
    
    return np.array(z_binned), np.array(mu_binned), np.array(sigma_binned)

if __name__ == "__main__":
    # Load your data
    # Assuming data_gold and data_silver are combined into a single dataset
    data = {
        "z": np.concatenate((np.linspace(0.01, 1.8, 20), np.linspace(0.1, 1.6, 15))),
        "mua0": np.concatenate((np.linspace(30, 45, 20), np.linspace(32, 42, 15))),
        "sigma_b": np.concatenate((np.ones(20) * 0.2, np.ones(15) * 0.3)),
    }

    z_data = data["z"]
    mu_data = data["mua0"]
    sigma_data = data["sigma_b"]

    # Bin the data with nΔz = 6
    n_bins = 6
    z_binned, mu_binned, sigma_binned = bin_data(z_data, mu_data, sigma_data, n_bins)

    # Compute distance moduli for ΛCDM model
    mu_LCDM = [mu(z, E_LCDM) for z in z_binned]

    # Compute residuals
    residuals = mu_binned - mu_LCDM

    # Main plot
    plt.figure(figsize=(10, 8))
    plt.errorbar(z_data, mu_data, yerr=sigma_data,
                 fmt='o', color='gray', label='All SNe Data', ecolor='lightgray', capsize=3)
    plt.plot(z_binned, mu_LCDM, label="ΛCDM (Ωm=0.27, ΩΛ=0.73)", color="blue", linestyle="--")
    plt.xlabel("Redshift (z)", fontsize=12)
    plt.ylabel("Distance Modulus μ", fontsize=12)
    plt.title("Distance Modulus vs Redshift", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)

    # Inset plot for residuals
    ax_inset = plt.axes([0.55, 0.15, 0.35, 0.35])  # [left, bottom, width, height]
    ax_inset.errorbar(z_binned, residuals, yerr=sigma_binned,
                      fmt='o', color='orange', label="Residuals", ecolor='gray', capsize=3)
    ax_inset.axhline(0, color='black', linestyle='--')
    ax_inset.set_xlabel("z", fontsize=10)
    ax_inset.set_ylabel("Δμ", fontsize=10)
    ax_inset.set_title("Binned Residuals (nΔz = 6)", fontsize=10)
    ax_inset.grid(True)

    # Show plot
    plt.show()
