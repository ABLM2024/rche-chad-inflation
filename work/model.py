"""Robust conformal Hedge ensemble (RCHE) for annual inflation in Chad: rolling-origin experiment.
All hyper-parameters are fixed a priori; nothing is tuned on the evaluation period."""
import numpy as np, pandas as pd, json, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from scipy import stats

CFG = dict(first=1997, eval_start=2005, eta=None, cap=10.0, alpha=0.2, gamma=0.05, threshold=3.0, winsor=False, label='base')
if len(sys.argv) > 1: CFG.update(json.loads(sys.argv[1]))
d = pd.read_csv('../data/chad.csv', index_col=0)
yr = d.index.values; y = d.infl.values.astype(float); u = d.unemp.values.astype(float)
lp = np.r_[0.0, np.cumsum(np.log1p(y/100))]            # log price level, lp[i+1] = end of year yr[i]

def gap_series(n):
    """Detrended log price level with a January-1994 level shift, fitted on levels up to end of yr[n-1]."""
    t = np.arange(n+1); D = (np.r_[yr[0]-1, yr[:n]] >= 1994).astype(float)
    X = np.c_[np.ones(n+1), t, D]; b = np.linalg.lstsq(X, lp[:n+1], rcond=None)[0]
    return 100*(lp[:n+1] - X@b)

def features(j, g, yy):
    return [yy[j-1], yy[j-2], g[j]]                      # g[j] = gap at end of yr[j-1]

LEARNERS = ['MED', 'HAR', 'LAH', 'SVR', 'GPR', 'RF']
def base_forecasts(i):
    yy = y.copy()
    if CFG['winsor']:                                   # robustness: winsorise the training history at its 5-95% quantiles
        lo, hi = np.percentile(y[:i], [5, 95]); yy[:i] = np.clip(y[:i], lo, hi)
    g = gap_series(i); tr = list(range(2, i))
    X = np.array([features(j, g, yy) for j in tr]); Y = yy[tr]; x = np.array([features(i, g, yy)])
    sc = StandardScaler().fit(X); Z = sc.transform(X); z = sc.transform(x)
    f = {}
    f['MED'] = np.median(yy[:i])
    f['HAR'] = HuberRegressor(epsilon=1.35, max_iter=500).fit(X[:, :2], Y).predict(x[:, :2])[0]
    f['LAH'] = HuberRegressor(epsilon=1.35, max_iter=500).fit(X[:, 2:], Y).predict(x[:, 2:])[0]
    f['SVR'] = SVR(kernel='rbf', C=5.0, epsilon=1.0, gamma='scale').fit(Z, Y).predict(z)[0]
    k = C(1.0)*RBF(np.ones(3), (0.1, 100.0)) + WhiteKernel(1.0, (1e-3, 1e3))
    gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=5, random_state=0).fit(Z, Y)
    m, s = gp.predict(z, return_std=True); f['GPR'] = m[0]; gps = s[0]
    f['RF'] = RandomForestRegressor(500, min_samples_leaf=3, random_state=0).fit(X, Y).predict(x)[0]
    # benchmarks
    b = {}
    b['RW'] = y[i-1]; b['Mean'] = y[:i].mean()
    ar = LinearRegression().fit(X[:, :2], Y); b['AR2'] = ar.predict(x[:, :2])[0]; ars = np.std(Y - ar.predict(X[:, :2]), ddof=3)
    a1 = LinearRegression().fit(X[:, :1], Y); b['AR1'] = a1.predict(x[:, :1])[0]
    return f, b, dict(gp_sd=gps, ar2_sd=ars, hist=y[:i].copy())

rows = []
idx = [i for i in range(len(y)) if yr[i] >= CFG['first']]
K = len(LEARNERS); T = len(idx)
eta = CFG['eta'] if CFG['eta'] else np.sqrt(8*np.log(K)/T)
L = np.zeros(K)                                         # cumulative normalised losses
res_ens = []; res_eq = []; res_mean = []; alpha_t = CFG['alpha']
for i in idx:
    f, b, aux = base_forecasts(i)
    fv = np.array([f[k] for k in LEARNERS])
    w = np.exp(-eta*(L - L.min())); w /= w.sum()
    ens = float(w@fv); eq = float(fv.mean()); medc = float(np.median(fv))
    row = dict(year=int(yr[i]), y=y[i], ens=ens, eq=eq, medcomb=medc, **{k: f[k] for k in LEARNERS}, **b,
               **{'w_'+k: w[j] for j, k in enumerate(LEARNERS)}, gp_sd=aux['gp_sd'], ar2_sd=aux['ar2_sd'], u=u[i], u_prev=u[i-1])
    # --- conformal predictive distribution and intervals from past out-of-sample residuals of the ensemble
    r = np.array(res_ens); n = len(r)
    row['n_cal'] = n
    if n >= 4:
        def cq(res, a):                                 # finite-sample conformal quantile of |residual|
            kq = int(np.ceil((len(res)+1)*(1-a))); kq = min(max(kq, 1), len(res))
            return np.sort(np.abs(res))[kq-1]
        for a, tag in ((CFG['alpha'], '80'), (0.1, '90')):
            q = cq(r, a); row[f'lo{tag}'] = ens - q; row[f'hi{tag}'] = ens + q
        qa = cq(r, min(max(alpha_t, 0.01), 0.99)); row['loACI'] = ens - qa; row['hiACI'] = ens + qa; row['alpha_t'] = alpha_t
        cpd = ens + r                                    # conformal predictive sample (Vovk-type, split residuals)
        row['p_above'] = (np.sum(cpd > CFG['threshold']) + 0.5)/(n + 1)
        row['p_defl'] = (np.sum(cpd < 0) + 0.5)/(n + 1)
        row['crps'] = float(np.mean(np.abs(cpd - y[i])) - 0.5*np.mean(np.abs(cpd[:, None] - cpd[None, :])))
        # conformal on the historical mean (benchmark)
        rm = np.array(res_mean); qm = cq(rm, CFG['alpha']); row['lo80_mean'] = b['Mean'] - qm; row['hi80_mean'] = b['Mean'] + qm
        cm = b['Mean'] + rm; row['crps_mean'] = float(np.mean(np.abs(cm - y[i])) - 0.5*np.mean(np.abs(cm[:, None] - cm[None, :])))
        row['p_above_mean'] = (np.sum(cm > CFG['threshold']) + 0.5)/(len(rm) + 1)
        err = int(not (row['loACI'] <= y[i] <= row['hiACI'])); alpha_t = alpha_t + CFG['gamma']*(CFG['alpha'] - err)
    # Gaussian benchmarks (AR(2) OLS and GP predictive)
    z80 = stats.norm.ppf(0.9)
    for nm, mu, sd in (('ar2g', b['AR2'], aux['ar2_sd']), ('gpg', f['GPR'], aux['gp_sd'])):
        row[f'lo80_{nm}'] = mu - z80*sd; row[f'hi80_{nm}'] = mu + z80*sd
        zz = (y[i]-mu)/sd; row[f'crps_{nm}'] = float(sd*(zz*(2*stats.norm.cdf(zz)-1) + 2*stats.norm.pdf(zz) - 1/np.sqrt(np.pi)))
        row[f'p_above_{nm}'] = 1 - stats.norm.cdf((CFG['threshold'] - mu)/sd)
    h = aux['hist']; ql, qh = np.quantile(h, [0.1, 0.9]); row['lo80_clim'] = ql; row['hi80_clim'] = qh
    row['crps_clim'] = float(np.mean(np.abs(h - y[i])) - 0.5*np.mean(np.abs(h[:, None] - h[None, :])))
    row['p_above_clim'] = float(np.mean(h > CFG['threshold']))
    rows.append(row)
    # update after observing y[i]
    loss = np.minimum(np.abs(fv - y[i])/CFG['cap'], 1.0); L += loss
    res_ens.append(y[i] - ens); res_mean.append(y[i] - b['Mean'])
P = pd.DataFrame(rows).set_index('year')
P.to_csv(f"pred_{CFG['label']}.csv")
print('eta = %.3f' % eta, 'forecasts', len(P), 'eval', (P.index >= CFG['eval_start']).sum())

# ---- ex-ante forecast for 2025 with all information up to 2024
i = len(y)
f, b, aux = base_forecasts(i)
fv = np.array([f[k] for k in LEARNERS]); w = np.exp(-eta*(L - L.min())); w /= w.sum(); ens = float(w@fv)
r = np.array(res_ens)
def cq(res, a):
    kq = int(np.ceil((len(res)+1)*(1-a))); kq = min(max(kq, 1), len(res)); return np.sort(np.abs(res))[kq-1]
cpd = ens + r
fc = dict(year=2025, ens=ens, lo80=ens-cq(r, 0.2), hi80=ens+cq(r, 0.2), lo90=ens-cq(r, 0.1), hi90=ens+cq(r, 0.1),
          loACI=ens-cq(r, min(max(alpha_t, 0.01), 0.99)), hiACI=ens+cq(r, min(max(alpha_t, 0.01), 0.99)), alpha_t=alpha_t,
          p_above=(np.sum(cpd > CFG['threshold'])+0.5)/(len(r)+1), p_defl=(np.sum(cpd < 0)+0.5)/(len(r)+1),
          base={k: float(v) for k, v in f.items()}, weights={k: float(x) for k, x in zip(LEARNERS, w)}, MED=float(np.median(y)), n_cal=len(r),
          misery_point=ens + u[-1])
json.dump(fc, open(f"fc2025_{CFG['label']}.json", 'w'), indent=1); print('2025', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in fc.items() if k not in ('base', 'weights')})
