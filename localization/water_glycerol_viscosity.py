"""
Using approach from to calculate viscosity https://doi.org/10.1021/ie071349z
Refractive index information from https://pubs.acs.org/doi/10.1021/ie50291a023
"""
import numpy as np
import matplotlib.pyplot as plt

# values @ 25C
# mu_h20 = 0.89e-3
# mu_glyc = 0.95
# rho_h20 = 1
# rho_glyc = 1.261
# cv = 0.9
cvs = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
Ts = np.array([17, 22.5, 25, 30]) + 273
# T = 22.5 + 273
kb = 1.38e-23
R = 50e-9



# convert volume fraction to mass fraction
def rho_glyc(T): return 1277 - 0.654 * (T - 273)
# see here for water parameterization http://ddbonline.ddbst.de/DIPPR105DensityCalculation/DIPPR105CalculationCGI.exe?component=Water
def rho_water(T): return 0.14395 / 0.0112**(1 + (1 - T / 649.727)**0.05107)
def cm_fn(cv, T): return cv * rho_glyc(T) / (cv * rho_glyc(T) + (1 - cv) * rho_water(T))
# viscosity of mixture
def visc_mix(cm, t): return visc_h20(t) ** alpha(cm, t) * visc_glyc(t) ** (1 - alpha(cm, t))
def visc_h20(t): return 1.79 * np.exp(-(1230 + (t - 273)) * (t-273) / (36100 + 360 * (t-273))) * 1e-2 / 10
def visc_glyc(t): return 12100 * np.exp(-(1233 - (t - 273)) * (t-273) / (9900 + 70*(t-273))) * 1e-2 / 10
def alpha(cm, t): return (1 - cm) + a(t) * b(t) * cm * (1-cm) / (a(t) * cm + b(t) * (1 - cm))
def a(t): return 0.705 - 0.0017 * (t - 273)
def b(t): return (4.9 + 0.036 * (t - 273)) * a(t)**2.5
# diffusion constant
def D(cm, t, R): return kb * t / (6 * np.pi * visc_mix(cm, t) * R)

# print(r"@cv=%.02f (cm=%0.2f) and t=%0.1f deg C, dynamic viscos=%0.3e Pa.s, R=%.0fnm: %0.3g um^2/s" %
#       (cv, cm_fn(cv, T), T - 273, visc_mix(cm_fn(cv, T), T), R * 1e9, D(cm_fn(cv, T), T, R) * 1e12))

cvs_all = np.linspace(0, 1, 100)
temps_all = np.linspace(20, 30, 100) + 273

figh = plt.figure(figsize=(16, 8))
grid = plt.GridSpec(2, 3, hspace=0.5, wspace=0.5)
plt.suptitle("Water/Glycerol mix viscosity versus temperature and concentration")

ax = plt.subplot(grid[0, 0])
for T in Ts:
    plt.semilogy(cvs_all, visc_mix(cm_fn(cvs_all, T), T))
plt.xlabel("Glycerol volume fraction")
plt.ylabel(r"Viscosity ($Pa \cdot s$)")
plt.title("Viscosity versus mix at fixed T")
plt.legend(["T = %0.2fC" % (T-273) for T in Ts])

ax = plt.subplot(grid[1, 0])
for T in Ts:
    plt.semilogy(cvs_all, D(cm_fn(cvs_all, T), T, R) * 1e12)
plt.xlabel("Glycerol volume fraction")
plt.ylabel("Diffusion constant ($\mu m^2/s$)")
plt.title("$D$ versus mix at fixed T\nradius %.0fnm beads" % (R * 1e9))

ax = plt.subplot(grid[0, 1])
for cv in cvs:
    plt.semilogy(temps_all - 273, visc_mix(cm_fn(cv, T), temps_all))
plt.xlabel("Temperature (C)")
plt.ylabel(r"Viscosity ($Pa \cdot s$)")
plt.title("Viscosity versus temp (fixed mix)" )
plt.legend(["Cv=%0.2f" % cv for cv in cvs])

ax = plt.subplot(grid[1, 1])
for cv in cvs:
    plt.semilogy(temps_all - 273, D(cm_fn(cv, temps_all), temps_all, R) * 1e12)
plt.xlabel("Temperature (C)")
plt.ylabel("Diffusion constant ($\mu m^2/s$)")
plt.title("$D$ versus T at fixed mix\nradius %.0fnm beads" % (R * 1e9))

ax = plt.subplot(grid[0, 2])
for T in Ts:
    ax.plot(cvs_all, cm_fn(cvs_all, T))
plt.xlabel("Volume fraction")
plt.ylabel("Mass fraction")
plt.title("Volume fraction vs. mass fraction at fixed T")
plt.legend(["T = %0.2fC" % (T-273) for T in Ts])

for volume_conc in cvs_all:
    print("%0.3f volume concentration, D=%0.3f $\mu^2/s$" % (volume_conc, D(cm_fn(volume_conc, T), T, R) * 1e12))