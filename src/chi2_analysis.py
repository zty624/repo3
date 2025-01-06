import numpy as np
import matplotlib.pyplot as plt

from data import *
from cosmos_model import *

if __name__ == "__main__":

    ### traditional model analysis
    omega_m_array = np.linspace(0.2, 0.7, 50)
    omega_l_array = np.linspace(0.1, 1, 50)
    # meshgrid
    Omega_m, Omega_l = np.meshgrid(omega_m_array, omega_l_array)
    chi2_LCDM = np.zeros_like(Omega_m)

    for i, omega_m in enumerate(omega_m_array):
        for j, omega_l in enumerate(omega_l_array):
            mu_LCDM   = np.array([mu(z, E_LCDM, omega_m, omega_l) for z in data["z"]])
            mu_residue = data["mua0"] - mu_LCDM
            chi2_LCDM[j, i] = np.sum(mu_residue ** 2 / (data["sigma_b"] ** 2 + data["Sigma_v_in_mu"] ** 2))
    
    # find minimum and plot 68%, 95% confidence interval
    min_index = np.unravel_index(np.argmin(chi2_LCDM), chi2_LCDM.shape)
    # confidence interval
    chi2_min = np.min(chi2_LCDM)
    chi2_68 = chi2_min + 1
    chi2_95 = chi2_min + 4
    chi2_99 = chi2_min + 9 

    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 16})
    plt.contourf(Omega_m, Omega_l, chi2_LCDM, levels=15)
    plt.colorbar()
    plt.plot(Omega_m[min_index], Omega_l[min_index], 'r*', label='min_chi2:%.2f' % (chi2_min))
    plt.plot(0.27, 0.73, 'y*', label='True Value')
    plt.contour(Omega_m, Omega_l, chi2_LCDM, levels=[chi2_68], colors='red', label='68%')
    plt.contour(Omega_m, Omega_l, chi2_LCDM, levels=[chi2_95], colors='yellow', label='95%')
    plt.contour(Omega_m, Omega_l, chi2_LCDM, levels=[chi2_99], colors='green', label='99%')
    plt.xlabel('$\Omega_m$')
    plt.ylabel('$\Omega_\Lambda$')
    plt.title('chi2_LCDM')
    plt.legend()
    plt.savefig('fig/chi2_LCDM.png')
    plt.show()


    ### dynamic model1 analysis: Omega_m, Omega_Lambda
    omega_m_array = np.linspace(0, 1, 50)
    omega_l_array = np.linspace(0, 1.5, 50)
    # meshgrid
    Omega_m, Omega_l = np.meshgrid(omega_m_array, omega_l_array)
    chi2_dynamic = np.zeros_like(Omega_m)

    for i, omega_m in enumerate(omega_m_array):
        for j, omega_l in enumerate(omega_l_array):
            mu_dymamic = np.array([mu_dynamic1(z, omega_m, omega_l) for z in data["z"]])
            mu_residue = data["mua0"] - mu_dymamic
            chi2_dynamic[j, i] = np.sum(mu_residue ** 2 / (data["sigma_b"] ** 2 + data["Sigma_v_in_mu"] ** 2))
    
    # find minimum and plot 68%, 95% confidence interval
    min_index = np.unravel_index(np.argmin(chi2_dynamic), chi2_dynamic.shape)
    # confidence interval
    chi2_min = np.min(chi2_dynamic)
    chi2_68 = chi2_min + 1
    chi2_95 = chi2_min + 4
    chi2_99 = chi2_min + 9

    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 16})
    plt.contourf(Omega_m, Omega_l, chi2_dynamic, levels=15)
    plt.colorbar()
    plt.plot(Omega_m[min_index], Omega_l[min_index], 'r*', label='min_chi2:%.2f' % (chi2_min))
    plt.plot(np.linspace(0, 1, 50), 1 - np.linspace(0, 1, 50), 'r--', label='Flat')
    plt.plot(0.27, 0.73, 'g*', label='True Value')
    plt.contour(Omega_m, Omega_l, chi2_dynamic, levels=[chi2_68], colors='red', label='68%')
    plt.contour(Omega_m, Omega_l, chi2_dynamic, levels=[chi2_95], colors='yellow', label='95%')
    plt.contour(Omega_m, Omega_l, chi2_dynamic, levels=[chi2_99], colors='green', label='99%')
    plt.xlabel('$\Omega_m$')
    plt.ylabel('$\Omega_\Lambda$')
    plt.title('chi2_dynamic')
    plt.legend()
    plt.savefig('fig/chi2_dynamic.png')
    plt.show()

    ### dynamic model2 analysis: Omega_m, w
    omega_m_array = np.linspace(0, 1, 50)
    w_array = np.linspace(-2, 0, 50)
    # meshgrid
    Omega_m, W_l = np.meshgrid(omega_m_array, w_array)
    chi2_dynamic2 = np.zeros_like(Omega_m)

    for i, omega_m in enumerate(omega_m_array):
        for j, w in enumerate(w_array):
            mu_dymamic = np.array([mu(z,E_dynamic2,omega_m, w) for z in data["z"]])
            mu_residue = data["mua0"] - mu_dymamic
            chi2_dynamic2[j, i] = np.sum(mu_residue ** 2 / (data["sigma_b"] ** 2 + data["Sigma_v_in_mu"] ** 2))
    
    # find minimum and plot 68%, 95% confidence interval
    min_index = np.unravel_index(np.argmin(chi2_dynamic2), chi2_dynamic2.shape)
    # confidence interval
    chi2_min = np.min(chi2_dynamic2)
    chi2_68 = chi2_min + 1
    chi2_95 = chi2_min + 4
    chi2_99 = chi2_min + 9

    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 16})
    plt.contourf(Omega_m, W_l, chi2_dynamic2, levels=15)
    plt.colorbar()
    plt.plot(Omega_m[min_index], W_l[min_index], 'r*', label='min_chi2:%.2f' % (chi2_min))
    plt.contour(Omega_m, W_l, chi2_dynamic2, levels=[chi2_68], colors='red', label='68%')
    plt.contour(Omega_m, W_l, chi2_dynamic2, levels=[chi2_95], colors='yellow', label='95%')
    plt.contour(Omega_m, W_l, chi2_dynamic2, levels=[chi2_99], colors='green', label='99%')
    plt.axhline(-1, color='red', linestyle='--', label='w=-1')
    plt.xlabel('$\Omega_m$')
    plt.ylabel('$w$')
    plt.title('chi2_dynamic2')
    plt.legend()
    plt.savefig('fig/chi2_dynamic2.png')
    plt.show()

    ### dynamic model3 analysis: w_0, w_a
    w0_array = np.linspace(-2.5, 0, 30)
    wa_array = np.linspace(-3, 8, 100)
    # meshgrid
    w_0, w_a = np.meshgrid(w0_array, wa_array)
    chi2_dynamic3 = np.zeros_like(w_0)

    for i, w0 in enumerate(w0_array):
        for j, wa in enumerate(wa_array):
            mu_dymamic = np.array([mu(z,E_dynamic3,w0, wa) for z in data["z"]])
            mu_residue = data["mua0"] - mu_dymamic
            chi2_dynamic3[j, i] = np.sum(mu_residue ** 2 / (data["sigma_b"] ** 2 + data["Sigma_v_in_mu"] ** 2))
    
    # find minimum and plot 68%, 95% confidence interval
    min_index = np.unravel_index(np.argmin(chi2_dynamic3), chi2_dynamic3.shape)
    # confidence interval
    chi2_min = np.min(chi2_dynamic3)
    chi2_68 = chi2_min + 1
    chi2_95 = chi2_min + 4
    chi2_99 = chi2_min + 9

    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 16})
    plt.contourf(w_0, w_a, chi2_dynamic3, levels=15)
    plt.colorbar()
    plt.plot(w_0[min_index], w_a[min_index], 'r*', label='min_chi2:%.2f' % (chi2_min))
    plt.plot(-1, 0, 'g*', label='True Value')
    plt.contour(w_0, w_a, chi2_dynamic3, levels=[chi2_68], colors='red', label='68%')
    plt.contour(w_0, w_a, chi2_dynamic3, levels=[chi2_95], colors='yellow', label='95%')
    plt.contour(w_0, w_a, chi2_dynamic3, levels=[chi2_99], colors='green', label='99%')
    plt.xlabel('$w_0$')
    plt.ylabel('$w_a$')
    plt.title('chi2_dynamic3')
    plt.legend()
    plt.savefig('fig/chi2_dynamic3.png')
    plt.show()

    ### extend model analysis: w_0, w_a
    w0_array = np.linspace(-3, 2, 50)
    wa_array = np.linspace(-4, 3, 60)
    # meshgrid
    w_0, w_a = np.meshgrid(w0_array, wa_array)
    chi2_dynamic4 = np.zeros_like(w_0)

    for i, w0 in enumerate(w0_array):
        for j, wa in enumerate(wa_array):
            mu_dymamic = np.array([mu(z,E_dynamic4,w0, wa) for z in data["z"]])
            mu_residue = data["mua0"] - mu_dymamic
            chi2_dynamic4[j, i] = np.sum(mu_residue ** 2 / (data["sigma_b"] ** 2 + data["Sigma_v_in_mu"] ** 2))
    
    # find minimum and plot 68%, 95% confidence interval
    min_index = np.unravel_index(np.argmin(chi2_dynamic4), chi2_dynamic4.shape)
    # confidence interval
    chi2_min = np.min(chi2_dynamic4)
    chi2_68 = chi2_min + 1
    chi2_95 = chi2_min + 4
    chi2_99 = chi2_min + 9

    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 16})
    plt.contourf(w_0, w_a, chi2_dynamic4, levels=15)
    plt.colorbar()
    plt.plot(w_0[min_index], w_a[min_index], 'r*', label='min_chi2:%.2f' % (chi2_min))
    plt.plot(-1, 0, 'g*', label='True Value')
    plt.contour(w_0, w_a, chi2_dynamic4, levels=[chi2_68], colors='red', label='68%')
    plt.contour(w_0, w_a, chi2_dynamic4, levels=[chi2_95], colors='yellow', label='95%')
    plt.contour(w_0, w_a, chi2_dynamic4, levels=[chi2_99], colors='green', label='99%')
    plt.xlabel('$w_0$')
    plt.ylabel('$w_a$')
    plt.title('chi2_dynamic4')
    plt.legend()
    plt.savefig('fig/chi2_dynamic4.png')
    plt.show()
