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
cv = 0.5
T = 22.5 + 273
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

print(r"@cv=%.02f (cm=%0.2f) and t=%0.1f deg C, dynamic viscos=%0.3e Pa.s, R=%.0fnm: %0.3g um^2/s" %
      (cv, cm_fn(cv, T), T - 273, visc_mix(cm_fn(cv, T), T), R * 1e9, D(cm_fn(cv, T), T, R) * 1e12))

cvs = np.linspace(0, 1, 100)
ts = np.linspace(20, 30, 100) + 273

figh = plt.figure(figsize=(16, 8))
grid = plt.GridSpec(2, 3, hspace=0.5, wspace=0.5)
plt.suptitle("Water/Glycerol mix viscosity versus temperature and concentration")

ax = plt.subplot(grid[0, 0])
plt.semilogy(cvs, visc_mix(cm_fn(cvs, T), T))
plt.xlabel("Glycerol volume fraction")
plt.ylabel(r"Viscosity ($Pa \cdot s$)")
plt.title("Versus mix @T = %0.1f C" % (T - 273))

ax = plt.subplot(grid[1, 0])
plt.semilogy(cvs, D(cm_fn(cvs, T), T, R) * 1e12)
plt.xlabel("Glycerol volume fraction")
plt.ylabel("Diffusion constant ($\mu m^2/s$)")
plt.title("%.0fnm bead diffusion" % (R * 1e9))

ax = plt.subplot(grid[0, 1])
plt.semilogy(ts - 273, visc_mix(cm_fn(cv, T), ts))
plt.xlabel("Temperature (C)")
plt.ylabel(r"Viscosity ($Pa \cdot s$)")
plt.title("Versus temp @Cv=%0.2f (Cm=%.2f)" % (cv, cm_fn(cv, T)))

ax = plt.subplot(grid[1, 1])
plt.semilogy(ts - 273, D(cm_fn(cv, ts), ts, R) * 1e12)
plt.xlabel("Temperature (C)")
plt.ylabel("Diffusion constant ($\mu m^2/s$)")
plt.title("%.0fnm bead diffusion" % (R * 1e9))

ax = plt.subplot(grid[0, 2])
ax.plot(cvs, cm_fn(cvs, T))
plt.xlabel("Volume fraction")
plt.ylabel("Mass fraction")
plt.title("Volume fraction vs. mass fraction @T = %0.1f C" % (T - 273))