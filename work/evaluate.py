"""Evaluation of the rolling-origin forecasts: point accuracy, Diebold-Mariano (HLN), model confidence set,
interval calibration, CRPS, event probabilities, misery index, data diagnostics."""
import numpy as np, pandas as pd, json, sys
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, kpss
lab = sys.argv[1] if len(sys.argv) > 1 else 'base'; ES = int(sys.argv[2]) if len(sys.argv) > 2 else 2005
P = pd.read_csv(f'pred_{lab}.csv', index_col=0); E = P.loc[ES:]
d = pd.read_csv('../data/chad.csv', index_col=0)
POINT = ['ens', 'eq', 'medcomb', 'MED', 'HAR', 'LAH', 'SVR', 'GPR', 'RF', 'RW', 'Mean', 'AR1', 'AR2']
e = E[POINT].sub(E.y, axis=0); out = {'label': lab, 'eval_start': ES, 'n': int(len(E))}
out['point'] = {m: dict(MAE=float(e[m].abs().mean()), RMSE=float(np.sqrt((e[m]**2).mean())),
                       MdAE=float(e[m].abs().median()), bias=float(-e[m].mean())) for m in POINT}
def dm_hln(e1, e2, h=1, power=1):
    dlt = np.abs(e1)**power - np.abs(e2)**power; n = len(dlt); db = dlt.mean()
    g0 = np.mean((dlt-db)**2); s = np.sqrt(g0/n)
    st = db/s*np.sqrt((n+1-2*h+h*(h-1)/n)/n)
    return float(st), float(2*stats.t.sf(abs(st), n-1))
out['dm_vs_MED'] = {m: dict(zip(['stat_abs', 'p_abs'], dm_hln(e[m].values, e['MED'].values))) | dict(zip(['stat_sq', 'p_sq'], dm_hln(e[m].values, e['MED'].values, power=2))) for m in POINT if m != 'MED'}
out['dm_vs_RW'] = {m: dict(zip(['stat_abs', 'p_abs'], dm_hln(e[m].values, e['RW'].values))) for m in POINT if m != 'RW'}
# model confidence set (Hansen, Lunde & Nason 2011), T_max statistic, stationary bootstrap
def mcs(Lmat, names, B=5000, blk=2.0, alpha=0.10, seed=0):
    rng = np.random.default_rng(seed); n = Lmat.shape[0]; M = list(range(Lmat.shape[1])); pv = {}; pmax = 0.0
    def boot_idx():
        idx = np.empty(n, int); idx[0] = rng.integers(n)
        for t in range(1, n): idx[t] = rng.integers(n) if rng.random() < 1/blk else (idx[t-1]+1) % n
        return idx
    IDX = [boot_idx() for _ in range(B)]
    while len(M) > 1:
        Lm = Lmat[:, M]; dbar = Lm.mean(0) - Lm.mean()
        bs = np.array([Lm[ix].mean(0) - Lm[ix].mean() for ix in IDX]) - dbar
        var = (bs**2).mean(0); tst = dbar/np.sqrt(var); Tmax = tst.max()
        Tb = (bs/np.sqrt(var)).max(1); p = float((Tb >= Tmax).mean()); pmax = max(pmax, p)
        worst = M[int(np.argmax(tst))]; pv[names[worst]] = pmax
        if pmax >= alpha: break
        M.remove(worst)
    for k in M: pv[names[k]] = 1.0 if names[k] not in pv else pv[names[k]]
    return pv, [names[k] for k in M]
for loss, pw in (('abs', 1), ('sq', 2)):
    pv, keep = mcs(np.abs(e.values)**pw, POINT); out[f'mcs_{loss}'] = dict(pvalues=pv, set90=keep)
# intervals and probabilistic scores
def winkler(lo, hi, yv, a):
    w = hi - lo; return float(np.mean(w + (2/a)*(lo - yv)*(yv < lo) + (2/a)*(yv - hi)*(yv > hi)))
IV = {'RCHE (80%)': ('lo80', 'hi80'), 'RCHE-ACI (80%)': ('loACI', 'hiACI'), 'Conformal mean (80%)': ('lo80_mean', 'hi80_mean'),
      'Gaussian AR(2) (80%)': ('lo80_ar2g', 'hi80_ar2g'), 'GP predictive (80%)': ('lo80_gpg', 'hi80_gpg'), 'Climatology (80%)': ('lo80_clim', 'hi80_clim')}
out['intervals'] = {k: dict(coverage=float(((E.y >= E[a]) & (E.y <= E[b])).mean()), width=float((E[b]-E[a]).mean()),
                            winkler=winkler(E[a].values, E[b].values, E.y.values, 0.2)) for k, (a, b) in IV.items()}
out['intervals']['RCHE (90%)'] = dict(coverage=float(((E.y >= E.lo90) & (E.y <= E.hi90)).mean()), width=float((E.hi90-E.lo90).mean()),
                                      winkler=winkler(E.lo90.values, E.hi90.values, E.y.values, 0.1))
out['crps'] = {k: float(E[c].mean()) for k, c in (('RCHE', 'crps'), ('Conformal mean', 'crps_mean'), ('Gaussian AR(2)', 'crps_ar2g'), ('GP predictive', 'crps_gpg'), ('Climatology', 'crps_clim'))}
ev = (E.y > 3).astype(float).values
base_rate_past = np.array([np.mean(d.infl.loc[:t-1].values > 3) for t in E.index])
br = lambda p: float(np.mean((p - ev)**2))
out['brier3'] = {k: br(E[c].values) for k, c in (('RCHE', 'p_above'), ('Conformal mean', 'p_above_mean'), ('Gaussian AR(2)', 'p_above_ar2g'), ('GP predictive', 'p_above_gpg'), ('Climatology', 'p_above_clim'))}
out['brier3']['Past frequency'] = br(base_rate_past); out['event_rate3'] = float(ev.mean())
evd = (E.y < 0).astype(float).values; out['brier_defl'] = {'RCHE': float(np.mean((E.p_defl - evd)**2)), 'Past frequency': float(np.mean((np.array([np.mean(d.infl.loc[:t-1].values < 0) for t in E.index]) - evd)**2))}
out['event_rate_defl'] = float(evd.mean())
# AUC of the 3% exceedance probability
from sklearn.metrics import roc_auc_score
out['auc3'] = {k: float(roc_auc_score(ev, E[c])) for k, c in (('RCHE', 'p_above'), ('Gaussian AR(2)', 'p_above_ar2g'), ('GP predictive', 'p_above_gpg'))}
# misery index (inflation forecast + random-walk unemployment forecast)
mi = E.y + E.u
for m in ('ens', 'MED', 'Mean', 'RW', 'AR2'):
    out.setdefault('misery', {})[m] = dict(MAE=float(np.abs(E[m] + E.u_prev - mi).mean()), RMSE=float(np.sqrt(((E[m] + E.u_prev - mi)**2).mean())))
out['unemp_rw_MAE'] = float(np.abs(E.u - E.u_prev).mean())
out['weights_final'] = {c[2:]: float(E[c].iloc[-1]) for c in E.columns if c.startswith('w_')}
out['weights_mean'] = {c[2:]: float(E[c].mean()) for c in E.columns if c.startswith('w_')}
# data diagnostics (full sample)
x = d.infl.values; x94 = d.infl.drop(1994).values
out['diag'] = dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), median=float(np.median(x)), sd_ex94=float(x94.std(ddof=1)), mean_ex94=float(x94.mean()),
                   skew=float(stats.skew(x)), kurt=float(stats.kurtosis(x)), skew_ex94=float(stats.skew(x94)),
                   acf=acf(x, nlags=4).tolist(), acf_ex94=acf(x94, nlags=4).tolist(),
                   lb_p=acorr_ljungbox(x, lags=[2, 4], return_df=True).lb_pvalue.tolist(), lb_p_ex94=acorr_ljungbox(x94, lags=[2, 4], return_df=True).lb_pvalue.tolist(),
                   adf_p=float(adfuller(x, autolag='AIC')[1]), share_above3=float(np.mean(x > 3)), share_neg=float(np.mean(x < 0)),
                   share_above3_since2005=float(np.mean(d.infl.loc[2005:].values > 3)), unemp_mean=float(d.unemp.mean()), unemp_acf1=float(acf(d.unemp.values, nlags=1)[1]),
                   jb_p=float(stats.jarque_bera(x).pvalue), jb_p_ex94=float(stats.jarque_bera(x94).pvalue))
json.dump(out, open(f'eval_{lab}_{ES}.json', 'w'), indent=1)
if __name__ == '__main__':
    print(pd.DataFrame(out['point']).T.sort_values('MAE').round(3).to_string())
    print('DM vs MED', {k: round(v['p_abs'], 3) for k, v in out['dm_vs_MED'].items()})
    print('MCS abs', out['mcs_abs']['set90'], '\nMCS sq', out['mcs_sq']['set90'])
    print(pd.DataFrame(out['intervals']).T.round(3).to_string()); print('CRPS', {k: round(v, 3) for k, v in out['crps'].items()})
    print('Brier3', {k: round(v, 3) for k, v in out['brier3'].items()}, 'rate', out['event_rate3'], 'AUC', out['auc3'])
    print('Brier defl', out['brier_defl'], out['event_rate_defl']); print('misery', out['misery'], out['unemp_rw_MAE'])
    print('weights', {k: round(v, 2) for k, v in out['weights_final'].items()})
