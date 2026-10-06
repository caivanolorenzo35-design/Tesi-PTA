#!/usr/bin/env python
"""
Analisi PTA di J1640+2224 con indice cromatico dello scattering LIBERO (sv_gp_alpha).

Versione script del notebook 08_Scattering_alpha_libero.ipynb, per girare da remoto
senza Jupyter. Fa solo il campionamento: i grafici restano nel notebook, che legge
la catena prodotta qui.

Dati di default: sim_data_noise_sv44/ (scattering generato con alpha = 4.4, senza GWB,
seed jax.random.key(1)).
Modello: 7 parametri = red_noise (2) + dm_gp (2) + sv_gp (2) + sv_gp_alpha.
Lo scattering usa ds.dmfourierbasis_alpha, quindi alpha e' un parametro campionato.

Esempi:
    python run_08_alpha_libero.py
    python run_08_alpha_libero.py --niter 2000000 --outdir ./test_sim_sv44_alphafree/
    nohup python run_08_alpha_libero.py > run_alpha_libero.log 2>&1 &
"""

import argparse
import os
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workdir', default=os.path.dirname(os.path.abspath(__file__)),
                   help='cartella della tesi, che contiene discovery_new/ e i dati (default: cartella dello script)')
    p.add_argument('--data', default='sim_data_noise_sv44/J1640+2224',
                   help='feather della pulsar da analizzare, relativo a workdir')
    p.add_argument('--outdir', default='./test_sim_sv44_alphafree/',
                   help='cartella di output del sampler')
    p.add_argument('--niter', type=int, default=1_000_000, help='numero di iterazioni PTMCMC')
    p.add_argument('--thin', type=int, default=10, help='thinning della catena')
    p.add_argument('--fref', type=float, default=1400.0,
                   help='frequenza di riferimento in MHz per la base dello scattering. '
                        'Con 1400 alpha e log10_A sono molto degeneri; un valore vicino alla '
                        'banda dei dati (es. 500) li scorrela, ma cambia il significato di log10_A')
    p.add_argument('--seed', type=int, default=None,
                   help='seed numpy per il punto iniziale estratto dalla prior (default: casuale)')
    p.add_argument('--no-resume', action='store_true',
                   help='non riprendere da una catena esistente in outdir')
    return p.parse_args()


def main():
    args = parse_args()

    os.chdir(args.workdir)
    sys.path.insert(0, './')          # per importare discovery_new dalla cartella della tesi

    import re

    import numpy as np
    import jax
    import jax.numpy as jnp
    from scipy.stats import uniform
    from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

    # usa discovery_new se presente nella cartella di lavoro (versione 0.5 usata nei notebook),
    # altrimenti il discovery installato nell'ambiente (es. 0.2 nel container di mithrandir)
    try:
        import discovery_new as ds
        _pkg = 'discovery_new'
    except ImportError:
        import discovery as ds
        _pkg = 'discovery'

    print('=' * 70, flush=True)
    print('Analisi con alpha LIBERO', flush=True)
    print(f'workdir : {os.getcwd()}', flush=True)
    print(f'dati    : {args.data}', flush=True)
    print(f'outdir  : {args.outdir}', flush=True)
    print(f'niter   : {args.niter}   thin: {args.thin}   fref: {args.fref} MHz', flush=True)
    print(f'jax     : {jax.__version__}   devices: {jax.devices()}', flush=True)
    print(f'x64     : {jax.config.x64_enabled}', flush=True)
    print(f'package : {_pkg} {getattr(ds, "__version__", "?")} ({ds.__file__})', flush=True)
    print('=' * 70, flush=True)

    if args.seed is not None:
        np.random.seed(args.seed)

    # ---------------------------------------------------------------- dati
    psr = ds.pulsar.Pulsar.read_feather(args.data)
    print(f'pulsar  : {psr.name}, {len(psr.toas)} TOA', flush=True)

    # ------------------------------------------------------------- modello
    # scattering con alpha libero: dmfourierbasis_alpha restituisce una funzione di alpha
    if args.fref == 1400.0:
        sv_basis = ds.dmfourierbasis_alpha
    else:
        def sv_basis(psr_, components, T=None, _fref=args.fref):
            return ds.dmfourierbasis_alpha(psr_, components, T=T, fref=_fref)

    model = [psr.residuals]
    model += [ds.makegp_timing(psr, svd=True)]
    model += [ds.makenoise_measurement(psr, psr.noisedict, tnequad=True, ecorr=True)]
    model.append(ds.makegp_fourier(psr, ds.powerlaw, psr.noisedict[psr.name + '_red_components'],
                                   T=ds.getspan(psr), name='red_noise'))
    model.append(ds.makegp_fourier(psr, ds.powerlaw, psr.noisedict[psr.name + '_dm_components'],
                                   T=ds.getspan(psr), name='dm_gp',
                                   fourierbasis=ds.make_dmfourierbasis(alpha=2.0, tndm=False)))
    model.append(ds.makegp_fourier(psr, ds.powerlaw, psr.noisedict[psr.name + '_chrom_components'],
                                   T=ds.getspan(psr), name='sv_gp',
                                   fourierbasis=sv_basis))

    logL = ds.PulsarLikelihood(model).logL
    print('parametri:', logL.params, flush=True)

    atteso = {psr.name + '_' + s for s in ['red_noise_gamma', 'red_noise_log10_A',
                                           'dm_gp_gamma', 'dm_gp_log10_A',
                                           'sv_gp_gamma', 'sv_gp_log10_A', 'sv_gp_alpha']}
    assert set(logL.params) == atteso, \
        f'Mancanti: {atteso - set(logL.params)}\nIn eccesso: {set(logL.params) - atteso}'
    print('OK: 7 parametri attesi, sv_gp_alpha compreso', flush=True)

    # -------------------------------------------------------------- prior
    my_priordict = {
        "(.*_)?sv_gp_gamma.*": [0, 7],
        "(.*_)?sv_gp_alpha.*": [0, 10],
        "(.*_)?sv_gp_log10_A.*": [-20, -11],
        "(.*_)?sw_log10_A.*": [-10, 1],
        "(.*_)?sw_gamma.*": [-5, 6],
        "(.*_)?dm_gp_gamma.*": [0, 7],
        "(.*_)?dm_gp_log10_A.*": [-20, -11],
        "(.*_)?red_noise_gamma.*": [0, 7],
        "(.*_)?red_noise_log10_A.*": [-20, -11],
        "(.*_)?sw_n_earth.*": [0, 30],
    }

    def build_priors(param_names, priordict):
        priors, unmatched = {}, []
        for name in param_names:
            for pattern, (lo, hi) in priordict.items():
                if re.fullmatch(pattern, name):
                    priors[name] = uniform(loc=lo, scale=hi - lo)
                    break
            else:
                unmatched.append(name)
        if unmatched:
            print(f'Attenzione: nessuna prior per {unmatched}, uso normale standard', flush=True)
        return priors

    priors = build_priors(logL.params, my_priordict)

    def params_dict_from_array(x):
        return dict(zip(logL.params, np.asarray(x)))

    def log_prior(x):
        logp = 0.0
        for name, value in params_dict_from_array(x).items():
            if name in priors:
                logp += priors[name].logpdf(value)
            else:
                logp += -0.5 * np.sum(np.asarray(value) ** 2)
        return logp

    def prior_draw():
        return np.array([priors[name].rvs() for name in logL.params])

    # --------------------------------------------------------- likelihood
    @jax.jit
    def _jit_logL(params_array):
        return logL(dict(zip(logL.params, params_array)))

    def log_likelihood(x):
        x = np.asarray(x, dtype=np.float64)
        return float(np.array(_jit_logL(jnp.array(x))))

    x0 = prior_draw()
    print('x0 =', x0, flush=True)
    print('logL iniziale =', log_likelihood(x0), flush=True)   # include la compilazione

    start = time.time()
    for _ in range(20):
        log_likelihood(x0)
    dt = (time.time() - start) / 20
    print(f'una valutazione = {dt * 1e3:.1f} ms  ->  {args.niter} iterazioni ~ '
          f'{dt * args.niter / 3600:.1f} ore (solo likelihood)', flush=True)

    # ------------------------------------------------------------ sampler
    os.makedirs(args.outdir, exist_ok=True)

    sampler = ptmcmc(
        ndim=len(logL.params),
        logl=log_likelihood,
        logp=log_prior,
        cov=np.eye(len(logL.params)) * 0.001,
        outDir=args.outdir,
        resume=not args.no_resume,
    )

    print(f'inizio sampling: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    t0 = time.time()
    sampler.sample(x0, Niter=args.niter, thin=args.thin)
    print(f'\nfine sampling: {time.strftime("%Y-%m-%d %H:%M:%S")} '
          f'({(time.time() - t0) / 3600:.2f} ore)', flush=True)

    chainfile = os.path.join(args.outdir, 'chain_1.txt')
    chain = np.loadtxt(chainfile)[:, :-4]
    chain = chain[len(chain) // 4:]
    print(f'catena: {chainfile}, {chain.shape[0]} campioni dopo il 25% di burn-in', flush=True)
    for i, name in enumerate(logL.params):
        q16, q50, q84 = np.percentile(chain[:, i], [16, 50, 84])
        print(f'  {name.replace(psr.name + "_", ""):20s} {q50:9.3f} [{q16:8.3f}, {q84:8.3f}]', flush=True)


if __name__ == '__main__':
    main()
