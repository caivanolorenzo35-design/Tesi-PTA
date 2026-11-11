"""Figura: deformazione di un anello di masse di prova per le polarizzazioni + e x.
L'onda si propaga lungo z (perpendicolare al foglio). Ampiezza esagerata per visibilita'.
Uso (dalla cartella Tesi_LaTeX):  python script/fig_polarizzazioni.py  -> Figure/cap1/polarizzazioni.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

h = 0.5                                   # ampiezza (esagerata: quella vera e' ~1e-15)
fasi = np.array([0, 0.5, 1, 1.5, 2]) * np.pi   # omega t
etichette = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
theta = np.linspace(0, 2*np.pi, 13)[:-1]  # 12 masse sull'anello
th_c = np.linspace(0, 2*np.pi, 300)       # contorno continuo

def deforma(x, y, pol, phi):
    c = 0.5 * h * np.cos(phi)
    if pol == "+":
        return x * (1 + c), y * (1 - c)
    else:                                  # x: stesso effetto ruotato di 45 gradi
        return x + c * y, y + c * x

fig, axes = plt.subplots(2, len(fasi), figsize=(10, 4.4))
for r, (pol, col, nome) in enumerate([("+", "#2E86AB", r"$h_+$"), ("x", "#C73E1D", r"$h_\times$")]):
    for k, phi in enumerate(fasi):
        ax = axes[r, k]
        ax.plot(np.cos(th_c), np.sin(th_c), color="0.8", lw=0.8, ls="--")   # anello a riposo
        xc, yc = deforma(np.cos(th_c), np.sin(th_c), pol, phi)
        ax.plot(xc, yc, color=col, lw=1.2)
        xm, ym = deforma(np.cos(theta), np.sin(theta), pol, phi)
        ax.plot(xm, ym, "o", color=col, ms=4)
        ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if r == 1:
            ax.set_xlabel(r"$\omega t = $" + etichette[k], fontsize=11)
        if k == 0:
            ax.set_ylabel(nome, fontsize=14, rotation=0, labelpad=18)
    # assi x, y indicativi nel primo pannello
    axes[r, 0].annotate("", xy=(1.4, -1.35), xytext=(0.9, -1.35), arrowprops=dict(arrowstyle="->", lw=0.8))
    axes[r, 0].annotate("", xy=(0.9, -0.85), xytext=(0.9, -1.35), arrowprops=dict(arrowstyle="->", lw=0.8))
    axes[r, 0].text(1.42, -1.35, "x", va="center", fontsize=9)
    axes[r, 0].text(0.9, -0.78, "y", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig("Figure/cap1/polarizzazioni.pdf")
