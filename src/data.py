import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

### Constants
C0 = 3e5                # light speed，km/s
H0 = 70                 # Hubble constant, km/s/Mpc
Omega_m = 0.27          # matter density parameter
Omega_Lambda = 0.73     # dark energy density parameter
Sigma_v = 400           # km/s

### data
data = pd.read_csv('src/data.csv')
data["dl"] = 10 ** ((data["mua0"]-25) / 5)                              # unit: mpc
data["sigma_dl"] = (data["dl"] / (5 * np.log(10))) * data["sigma_b"]
data["comoving_distance"] = 1/2997.9/(1+data["z"]) * 10 ** (data['mua0']/5 - 5)
data["Sigma_v_in_mu"] = 5 * Sigma_v / H0 / np.log(10) / np.power(10, data["mua0"]/5 - 5)

data_gold   = data[data["sample"] == "Gold"]
data_silver = data[data["sample"] == "Silver"]


if __name__ == "__main__":
    plt.figure(figsize=(10,8))
    plt.rcParams.update({'font.size': 16})
    plt.errorbar(data_gold["z"], data_gold["dl"], yerr=data_gold["sigma_dl"],
                fmt='*', color='green', label='SN', ecolor='red', capsize=3)
    plt.errorbar(data_silver["z"], data_silver["dl"], yerr=data_silver["sigma_dl"],
                fmt='*', color='blue', label='SN', ecolor='red', capsize=3)
    plt.legend(['Gold', 'Silver'])
    plt.grid(True)
    plt.xlabel('Redshift (z)')
    plt.ylabel('Luminosity Distance (d_L) [Mpc]')
    plt.xlim(0,2)
    plt.title('Distance-Redshift for SNe Ia over z', fontsize=20)
    plt.savefig('fig/SN_scatter.png')
    # plt.show()

    plt.figure(figsize=(10,8))
    plt.rcParams.update({'font.size': 16})
    plt.errorbar(data_gold["z"], data_gold["mua0"], yerr=data_gold["sigma_b"],
                fmt='*', color='green', label='SN', ecolor='red', capsize=3)
    plt.errorbar(data_silver["z"], data_silver["mua0"], yerr=data_silver["sigma_b"],
                fmt='*', color='blue', label='SN', ecolor='red', capsize=3)
    plt.legend(['Gold', 'Silver'])
    plt.grid(True)
    plt.xlabel('Redshift (z)')
    plt.ylabel('Luminosity Distance (mu)')
    plt.xlim(0,2)
    plt.title('Distance-Redshift for SNe Ia over z', fontsize=20)
    plt.savefig('fig/SN_scatter_1.png')
    # plt.show()

    plt.figure(figsize=(10,8))
    plt.rcParams.update({'font.size': 16})
    plt.scatter(data_gold["z"], data_gold["comoving_distance"], color='green', label='Gold')
    plt.scatter(data_silver["z"], data_silver["comoving_distance"], color='blue', label='Silver')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Redshift (z)')
    plt.ylabel('Comoving Distance (Mpc)')
    plt.xlim(0,2)
    plt.title('Distance-Redshift for SNe Ia over z', fontsize=20)
    plt.savefig('fig/SN_scatter_2.png')
    plt.show()
