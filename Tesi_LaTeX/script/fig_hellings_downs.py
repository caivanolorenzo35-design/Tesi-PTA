"""Figura della curva di Hellings & Downs (Capitolo 2).
Uso:  python script/fig_hellings_downs.py   (dalla cartella Tesi_LaTeX)
"""
import numpy as np
import matplotlib.pyplot as plt

zeta = np.linspace(1e-6, np.pi, 500)            # separazione angolare [rad]
x = (1 - np.cos(zeta)) / 2
gamma_hd = 0.5 - x / 4 + 1.5 * x * np.log(x)    # Eq. (HD) del Capitolo 2

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.degrees(zeta), gamma_hd, color="#2E86AB", lw=2)
ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.set_xlabel(r"Separazione angolare $\zeta$ [gradi]")
ax.set_ylabel(r"Correlazione $\Gamma(\zeta)$")
ax.set_xlim(0, 180)
ax.set_xticks(range(0, 181, 30))
fig.tight_layout()
fig.savefig("Figure/cap2/hellings_downs.pdf")
print("min Gamma = %.3f a %.1f gradi; Gamma(180) = %.3f"
      % (gamma_hd.min(), np.degrees(zeta[gamma_hd.argmin()]), gamma_hd[-1]))
