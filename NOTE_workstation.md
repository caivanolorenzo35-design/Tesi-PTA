# Workstation mithrandir — accesso e run

Appunti operativi. Accesso ottenuto a settembre 2026 tramite Irene e Golam Shaifullah.
Ultimo aggiornamento: 20 ottobre 2026.

---

## Coordinate

| | |
|---|---|
| macchina | `mithrandir` (UniMiB) |
| indirizzo | `149.132.54.143` |
| account | `lorenzo.caivano` (**con il punto**, diverso dall'utente Mac `lorenzocaivano`) |
| home | `/homes/lorenzo.caivano` |
| cartella di lavoro | `/homes/lorenzo.caivano/Tesi` (cioè `~/Tesi`) |
| container singularity | `/homes/Containers/discovery-cuda12.sif` |
| python nel container | `/opt/conda/envs/discovery-gpu-env/bin/python` |
| cartella Tesi sul Mac | `~/Desktop/Anno3Fisica/Tesi` |

La password non è scritta qui: è quella impostata al cambio forzato del primo accesso.

---

## 1. Accesso

```bash
ssh lorenzo.caivano@149.132.54.143
cd ~/Tesi
```

Sessioni tmux (le run devono girare dentro tmux, altrimenti muoiono quando chiudi il terminale):

```bash
tmux ls                      # elenca le sessioni esistenti
tmux attach -t <nome>        # rientra in una sessione
tmux new -s <nome>           # ne crea una nuova
tmux display-message -p '#S' # nome della sessione in cui sei
echo $TMUX                   # vuoto = NON sei in tmux
```

Staccarsi senza fermare la run: `Ctrl+B`, lasciare, poi `D`.

---

## 2. Prerequisiti in `~/Tesi`

Prima di lanciare qualsiasi cosa devono esserci:

1. `run_08_alpha_libero.py` — lo script generico (vale per tutte le analisi a 7 parametri con alpha libero, non solo il notebook 08: cambiano solo `--data` e `--outdir`)
2. `discovery_new/` — **indispensabile**. Lo script fa `sys.path.insert(0, './')` e quindi la trova per prima; il discovery installato nel container è la 0.2 e crasha in `makegp_ecorr` con `ValueError: need at least one array to concatenate`. La 0.5 locale ha i guard che servono.
3. la cartella dei dati che passi con `--data`

Per questo lo script va lanciato **da dentro `~/Tesi`**, mai da altrove: usa percorsi relativi.

---

## 3. Copiare i dati dal Mac alla workstation

`rsync` non crea i livelli intermedi, quindi prima la cartella va creata a mano sulla workstation:

```bash
# su mithrandir
mkdir -p ~/Tesi/dati/DR2low_feathers
```

```bash
# sul Mac, in una finestra di Terminale a parte
rsync -av ~/Desktop/Anno3Fisica/Tesi/dati/DR2low_feathers/J1640+2224 \
  lorenzo.caivano@149.132.54.143:/homes/lorenzo.caivano/Tesi/dati/DR2low_feathers/
```

Il feather dei dati reali è un singolo file di 4278786 byte.

---

## 4. Lanciare una run

```bash
tmux new -s <nome_sessione>
cd ~/Tesi

singularity exec /homes/Containers/discovery-cuda12.sif \
  python run_08_alpha_libero.py \
  --data ./dati/DR2low_feathers/J1640+2224 \
  --outdir ./test_real_alphafree/ \
  > run_reali_alphafree.log 2>&1 &

tail -f run_reali_alphafree.log
```

Note:

- la `&` finale manda il processo in background; dentro tmux **non serve** `nohup`
- `Ctrl+C` ferma solo il `tail`, non la run
- `singularity exec` senza `--nv` non espone la GPU: JAX gira su CPU (`devices: [CpuDevice(id=0)]`). Tutte le run fatte finora sono CPU, quindi per restare confrontabili va lasciato così
- i messaggi `not a valid test operator` all'avvio vengono dagli script di startup del container: innocui
- lo script ha `resume=True`: se la run si interrompe, rilanciandola con lo stesso `--outdir` riprende da dove era

### Opzioni dello script

```
--data     cartella/file del feather          (default: sim_data_noise_sv44/J1640+2224)
--outdir   cartella di output della catena    (default: ./test_sim_sv44_alphafree/)
--niter    iterazioni PTMCMC                  (default: 1000000)
--thin     thinning                           (default: 10)
--fref     frequenza di riferimento in MHz    (default: 1400.0)
--seed     seme                               (default: None)
--no-resume
```

Attenzione a `--fref`: il prior su `sv_gp_log10_A` nello script è fisso a `[-20, -11]`, tarato per fref = 1400.
Con un fref diverso l'ampiezza trasla di `alpha * log10(1400/fref)` e il prior va ricentrato a mano.

### Cosa deve stampare il log nei primi minuti

```
package : discovery_new 0.5 (...)        <- deve essere 0.5, non 0.2
pulsar  : J1640+2224, 6777 TOA           <- 6777 = dati reali DR2low
OK: 7 parametri attesi, sv_gp_alpha compreso
una valutazione = XXX ms -> 1000000 iterazioni ~ NN ore
```

---

## 5. Controllare l'avanzamento

```bash
ssh lorenzo.caivano@149.132.54.143
wc -l ~/Tesi/test_real_alphafree/chain_1.txt    # con niter=1e6 e thin=10 arriva a 100000
tail -3 ~/Tesi/run_reali_alphafree.log
ps -u lorenzo.caivano -f | grep run_08_alpha_libero | grep -v grep
```

Il tempo CPU nel `ps` cresce più dell'orologio perché BLAS è multithread: normale, non è un errore.

---

## 6. Riportare la catena sul Mac

```bash
# sul Mac
rsync -av lorenzo.caivano@149.132.54.143:/homes/lorenzo.caivano/Tesi/test_real_alphafree/ \
  ~/Desktop/Anno3Fisica/Tesi/test_real_alphafree_ws/
```

Convenzione: il suffisso `_ws` in locale marca "catena completa prodotta sulla workstation", per non
sovrascrivere eventuali catene parziali fatte sul Mac. I notebook leggono le cartelle `_ws`.

---

## 7. Run fatte su mithrandir

| outdir | log | contenuto |
|---|---|---|
| `test_sim_sv44_alphafree` | `run_base_ws.log` | simulazione con scattering alpha=4.4, nessun GWB, analisi con alpha libero (notebook 08) |
| `test_gwb_A140_sv44_alphafree` | `run_gwb140_ws.log` | GWB log10A=-14.0, gamma=13/3, su sv44, alpha libero (notebook 13) |
| `test_gwb_A1405_sv44_alphafree` | `run_gwb1405_ws.log` | GWB log10A=-14.05, gamma=13/3, su sv44, alpha libero (notebook 14) |
| `test_real_alphafree` | `run_reali_alphafree.log` | **dati reali** DR2low, alpha libero, fref=1400 (notebook 15) |

L'ultima è partita il 20/10/2026 alle 17:30 nella sessione tmux `reali`, stima 39.7 ore di sola
likelihood, fine prevista giovedì 22 ottobre in mattinata.
