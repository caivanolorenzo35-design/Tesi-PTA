"""Figura dei residui di timing di PSR J1640+2224 (Capitolo 2).
Dati: EPTA DR2low (file feather in ../dati/DR2low_feathers/).
Uso:  python script/fig_residuals_J1640.py   (dalla cartella Tesi_LaTeX)
"""
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.feather as feather

# lettura dei dati: tempi di arrivo [s], errori [s], residui [s], frequenze radio [MHz]
dati = feather.read_table("../dati/DR2low_feathers/J1640+2224",
                          columns=["toas", "toaerrs", "residuals", "freqs"]).to_pandas()

mjd = dati["toas"] / 86400.0                     # da secondi a MJD
anno = 2000.0 + (mjd - 51544.5) / 365.25         # da MJD ad anno
res = dati["residuals"] * 1e6                    # residui in microsecondi
err = dati["toaerrs"] * 1e6                      # errori in microsecondi
freq = dati["freqs"]                             # frequenza radio in MHz

bassa = freq < 500                               # LOFAR (circa 120-190 MHz)
alta = freq >= 500                               # telescopi EPTA (1.3-2.6 GHz)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 3.6), sharex=True)

ax1.errorbar(anno[bassa], res[bassa], yerr=err[bassa], fmt=".", ms=2,
             color="#2E86AB", ecolor="#2E86AB", elinewidth=0.5, alpha=0.5)
ax1.axhline(0, color="grey", lw=0.8, ls="--")
ax1.set_ylabel(r"Residuals [$\mu$s]")
ax1.set_title("LOFAR, 120–190 MHz", fontsize=10)

ax2.errorbar(anno[alta], res[alta], yerr=err[alta], fmt=".", ms=2,
             color="#2E86AB", ecolor="#2E86AB", elinewidth=0.5, alpha=0.5)
ax2.axhline(0, color="grey", lw=0.8, ls="--")
ax2.set_ylabel(r"Residuals [$\mu$s]")
ax2.set_xlabel("Year")
ax2.set_title("EPTA telescopes, 1.3–2.6 GHz", fontsize=10)

fig.tight_layout()
fig.savefig("Figure/cap2/residuals_J1640.pdf")
print("TOA totali:", len(dati), " bassa freq:", bassa.sum(), " alta freq:", alta.sum())
print("rms bassa: %.1f us, rms alta: %.1f us" % (res[bassa].std(), res[alta].std()))
