"""Figures (14 pt bold, 600 dpi, width W)."""
import numpy as np, pandas as pd, json, os, sys
from style14 import plt, W
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from statsmodels.tsa.stattools import acf
os.makedirs('figs', exist_ok=True)
d = pd.read_csv('../data/chad.csv', index_col=0)
P = pd.read_csv('pred_base.csv', index_col=0); E = P.loc[2005:]
O = json.load(open('eval_base_2005.json')); OW = json.load(open('eval_winsor_2005.json'))
FC = json.load(open('fc2025_base.json'))
LEARN = ['MED', 'HAR', 'LAH', 'SVR', 'GPR', 'RF']
LC = {'MED': '#2a78d6', 'HAR': '#1baf7a', 'LAH': '#eda100', 'SVR': '#eb6834', 'GPR': '#9a62d6', 'RF': '#8a8a84'}
ONLY = sys.argv[1:]
def want(n): return not ONLY or n in ONLY

if want('1'):
    fig, ax = plt.subplots(figsize=(W, 7.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 10.6); ax.axis('off')
    def box(x, y, w, h, t, fc):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.02,rounding_size=0.1', fc=fc, ec='#333330', lw=1.5))
        ax.text(x, y, t, ha='center', va='center', fontsize=14, fontweight='bold', linespacing=1.15)
    def arr(x0, y0, x1, y1): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=18, lw=2, color='#333330'))
    box(5, 9.9, 9.6, 0.9, 'Inflation and unemployment, 1991–2024', '#ececea')
    box(5, 8.5, 9.6, 0.9, 'Features: lags + level gap (1994 shift)', '#fde9b8')
    xs = [0.85, 2.51, 4.17, 5.83, 7.49, 9.15]
    for x, k in zip(xs, LEARN): box(x, 6.95, 1.5, 0.95, k, '#dce9f8')
    arr(5, 8.03, 5, 7.47)
    box(5, 5.3, 9.6, 0.9, 'Online Hedge weights (absolute loss)', '#d6f0e5')
    for x in xs: arr(x, 6.45, x, 5.78)
    box(5, 3.8, 9.6, 0.9, 'Conformal predictive distribution', '#f3dbe6')
    box(5, 2.3, 9.6, 0.9, 'ACI intervals, CRPS, P(π > 3%)', '#e4e0f7')
    box(5, 0.8, 9.6, 0.9, 'Rolling-origin evaluation, DM, MCS', '#ececea')
    for y0 in (9.43, 4.83, 3.33, 1.83): arr(5, y0, 5, y0-0.56)
    fig.savefig('figs/fig1_framework.png'); plt.close(fig)

if want('2'):
    fig, axs = plt.subplots(2, 1, figsize=(W, 7.4), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    ax = axs[0]; yrs = d.index.values
    ax.bar(yrs, d.infl, color=np.where(d.infl > 3, '#eb6834', np.where(d.infl < 0, '#2a78d6', '#9a9a94')), width=0.8)
    ax.plot(yrs, d.misery, color='#1f1f1c', marker='o', ms=4, lw=1.6, label='Misery index')
    ax.axhline(3, color='#d0453a', ls='--', lw=1.8); ax.text(1989.9, 4.4, '3%', color='#d0453a', ha='left', fontsize=14, fontweight='bold')
    ax.annotate('CFA franc\ndevaluation', (1994, 41.3), xytext=(1997.5, 20), fontsize=14, fontweight='bold', arrowprops=dict(arrowstyle='-|>', lw=1.5, color='#333330'))
    ax.set_ylabel('Rate (%)'); ax.set_title('(a) Inflation (bars) and misery index', loc='left'); ax.set_ylim(-14, 46)
    ax.legend(handles=[Patch(color='#eb6834', label='> 3%'), Patch(color='#9a9a94', label='0–3%'), Patch(color='#2a78d6', label='< 0%'), Line2D([], [], color='#1f1f1c', marker='o', ms=4, label='Misery index')],
              loc='upper right', ncol=2, columnspacing=0.8, handlelength=1.2)
    axs[1].plot(yrs, d.unemp, color='#1baf7a', marker='o', ms=5); axs[1].set_ylabel('Unemp. (%)'); axs[1].set_title('(b) Unemployment', loc='left'); axs[1].set_ylim(0.4, 2.0)
    fig.tight_layout(); fig.savefig('figs/fig2_data.png'); plt.close(fig)

if want('3'):
    fig, axs = plt.subplots(2, 1, figsize=(W, 7.6))
    x = d.infl.values; x94 = d.infl.drop(1994).values; lag = np.arange(1, 7); n = len(x)
    a1 = acf(x, nlags=6)[1:]; a2 = acf(x94, nlags=6)[1:]
    axs[0].bar(lag-0.18, a1, width=0.34, color='#9a9a94', label='Full sample', zorder=3); axs[0].bar(lag+0.18, a2, width=0.34, color='#2a78d6', label='Excluding 1994', zorder=3)
    cb = 1.96/np.sqrt(n); axs[0].axhspan(-cb, cb, color='#ececea', alpha=0.8, lw=0, zorder=1); axs[0].axhline(0, color='#333330', lw=1)
    axs[0].set_xlabel('Lag (years)'); axs[0].set_ylabel('ACF'); axs[0].set_title('(a) Autocorrelation of inflation', loc='left'); axs[0].legend(loc='lower right'); axs[0].set_ylim(-0.6, 0.4)
    lp = 100*np.r_[0, np.cumsum(np.log1p(x/100))]; yy = np.r_[1990, d.index.values]; t = np.arange(len(lp)); D = (yy >= 1994).astype(float)
    X = np.c_[np.ones(len(lp)), t, D]; b = np.linalg.lstsq(X, lp, rcond=None)[0]
    axs[1].plot(yy, lp, color='#1f1f1c', marker='o', ms=4, label='Log price level'); axs[1].plot(yy, X@b, color='#eb6834', ls='--', label='Trend + 1994 shift')
    ax2 = axs[1].twinx(); ax2.bar(yy, lp - X@b, color='#2a78d6', alpha=0.35, width=0.8); ax2.set_ylabel('Gap (%)'); ax2.set_ylim(-30, 30); ax2.spines['right'].set_visible(True); ax2.grid(False)
    axs[1].set_ylabel('100 × log level'); axs[1].set_title('(b) Price level, trend and gap (bars)', loc='left'); axs[1].legend(loc='upper left')
    fig.tight_layout(); fig.savefig('figs/fig3_diagnostics.png'); plt.close(fig)

if want('4'):
    fig, ax = plt.subplots(figsize=(W, 4.8)); yrs = E.index.values
    ax.fill_between(yrs, E.loACI, E.hiACI, color='#2a78d6', alpha=0.18, lw=0, label='RCHE 80% (ACI)')
    ax.plot(yrs, E.y, color='#1f1f1c', marker='o', ms=6, label='Observed'); ax.plot(yrs, E.ens, color='#2a78d6', marker='s', ms=5, label='RCHE')
    ax.plot(yrs, E.MED, color='#eb6834', ls='--', lw=2, label='Median')
    ax.errorbar([2025], [FC['ens']], yerr=[[FC['ens']-FC['loACI']], [FC['hiACI']-FC['ens']]], fmt='D', color='#d0453a', ms=8, capsize=5, lw=2.2, label='2025 forecast')
    ax.axhline(3, color='#d0453a', ls=':', lw=1.6); ax.set_ylabel('Inflation (%)'); ax.set_xlim(2004.3, 2025.7); ax.set_ylim(-18, 22)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=0.6, handlelength=1.3)
    fig.savefig('figs/fig4_forecasts.png'); plt.close(fig)

if want('5'):
    fig, ax = plt.subplots(figsize=(W, 4.6)); yrs = P.index.values
    Wt = np.array([P['w_'+k] for k in LEARN])
    ax.stackplot(yrs, Wt, colors=[LC[k] for k in LEARN], labels=LEARN, alpha=0.9, edgecolor='white', lw=0.6)
    ax.axvline(2004.5, color='#1f1f1c', ls='--', lw=1.6); ax.text(2005.0, 0.08, 'evaluation →', fontsize=14, fontweight='bold', va='bottom', color='white')
    ax.set_ylabel('Hedge weight'); ax.set_xlim(yrs[0], yrs[-1]); ax.set_ylim(0, 1)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=6, columnspacing=0.5, handlelength=1.0, handletextpad=0.3)
    fig.savefig('figs/fig5_weights.png'); plt.close(fig)

if want('6'):
    NAMES = {'ens': 'RCHE', 'eq': 'Equal-weight', 'medcomb': 'Median comb.', 'MED': 'Median', 'HAR': 'Huber AR(2)', 'LAH': 'Level-gap Huber', 'SVR': 'SVR', 'GPR': 'GP', 'RF': 'Random forest',
             'RW': 'Random walk', 'Mean': 'Mean', 'AR1': 'AR(1)', 'AR2': 'AR(2)'}
    pt = pd.DataFrame(O['point']).T.sort_values('MAE', ascending=False); ptw = pd.DataFrame(OW['point']).T
    fig, ax = plt.subplots(figsize=(W, 6.4)); yy = np.arange(len(pt))
    ax.barh(yy+0.2, pt.MAE, height=0.38, color='#2a78d6', label='Raw history')
    ax.barh(yy-0.2, ptw.loc[pt.index, 'MAE'], height=0.38, color='#eda100', label='Winsorised history')
    for k, m in enumerate(pt.index):
        if m != 'MED' and O['dm_vs_MED'][m]['p_abs'] < 0.05: ax.text(pt.MAE[m]+0.1, k+0.2, '*', va='center', fontsize=16, fontweight='bold')
    ax.set_yticks(yy, [NAMES[m] for m in pt.index]); ax.set_xlabel('MAE, 2005–2024 (pp)'); ax.set_xlim(0, 7.6)
    ax.legend(loc='lower center', bbox_to_anchor=(0.4, 1.0), ncol=2, columnspacing=0.8); ax.grid(axis='y', visible=False)
    fig.savefig('figs/fig6_point.png'); plt.close(fig)

if want('7'):
    I = O['intervals']; keys = ['RCHE-ACI (80%)', 'RCHE (80%)', 'Conformal mean (80%)', 'Climatology (80%)', 'Gaussian AR(2) (80%)', 'GP predictive (80%)']
    lab = ['RCHE-ACI', 'RCHE', 'Conf. mean', 'Climatology', 'Gauss. AR(2)', 'GP']; cols = ['#2a78d6', '#7fb0ea', '#eda100', '#8a8a84', '#1baf7a', '#9a62d6']
    fig, axs = plt.subplots(2, 1, figsize=(W, 8.0))
    for k, l, c in zip(keys, lab, cols):
        axs[0].scatter(I[k]['width'], 100*I[k]['coverage'], s=160, color=c, edgecolor='#1f1f1c', lw=1.0, zorder=3, label=l)
    axs[0].axhline(80, color='#d0453a', ls='--', lw=1.8); axs[0].set_xlabel('Mean width (pp)'); axs[0].set_ylabel('Coverage (%)'); axs[0].set_ylim(55, 105); axs[0].set_xlim(10, 31)
    axs[0].set_title('(a) 80% intervals: coverage vs width', loc='left'); axs[0].legend(loc='lower right', ncol=2, columnspacing=0.5, handletextpad=0.2, fontsize=14)
    cr = O['crps']; ck = ['RCHE', 'Climatology', 'Conformal mean', 'Gaussian AR(2)', 'GP predictive']; cl = ['RCHE', 'Clim.', 'Conf.\nmean', 'Gauss.\nAR(2)', 'GP']
    axs[1].bar(range(5), [cr[k] for k in ck], color=['#2a78d6', '#8a8a84', '#eda100', '#1baf7a', '#9a62d6'])
    for k, v in enumerate([cr[x] for x in ck]): axs[1].text(k, v+0.05, f'{v:.2f}', ha='center', fontsize=14, fontweight='bold')
    axs[1].set_xticks(range(5), cl); axs[1].set_ylabel('Mean CRPS (pp)'); axs[1].set_ylim(0, 5.2); axs[1].set_title('(b) Continuous ranked probability score', loc='left')
    fig.tight_layout(); fig.savefig('figs/fig7_probabilistic.png'); plt.close(fig)

if want('8'):
    fig, ax = plt.subplots(figsize=(W, 4.4)); yrs = E.index.values; ev = E.y > 3
    ax.bar(yrs, E.p_above, color=np.where(ev, '#eb6834', '#2a78d6'), width=0.75)
    ax.axhline(0.5, color='#1f1f1c', ls='--', lw=1.4); ax.set_ylabel('P(π > 3%)'); ax.set_ylim(0, 1); ax.set_xlim(2004.3, 2024.7)
    ax.legend(handles=[Patch(color='#eb6834', label='Observed > 3%'), Patch(color='#2a78d6', label='Observed ≤ 3%')], loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.savefig('figs/fig8_exceedance.png'); plt.close(fig)

if want('9'):
    cases = [('Base', 'base'), ('η × 0.5', 'eta05'), ('η × 2', 'eta2'), ('Cap 5 pp', 'cap5'), ('Cap 20 pp', 'cap20'), ('γ = 0.10', 'gamma10'), ('Winsorised', 'winsor')]
    R = {lab: json.load(open(f'eval_{k}_2005.json')) for lab, k in cases}
    fig, axs = plt.subplots(3, 1, figsize=(W, 8.6), sharex=True); xx = np.arange(len(cases))
    axs[0].bar(xx, [R[c]['point']['ens']['MAE'] for c, _ in cases], color='#2a78d6'); axs[0].axhline(R['Base']['point']['MED']['MAE'], color='#eb6834', ls='--', lw=2, label='Median')
    axs[0].set_ylabel('MAE (pp)'); axs[0].set_ylim(3, 4.6); axs[0].legend(loc='upper right')
    axs[1].bar(xx, [R[c]['crps']['RCHE'] for c, _ in cases], color='#1baf7a'); axs[1].axhline(R['Base']['crps']['Climatology'], color='#8a8a84', ls='--', lw=2, label='Climatology')
    axs[1].set_ylabel('CRPS (pp)'); axs[1].set_ylim(2.5, 3.1); axs[1].legend(loc='upper right')
    axs[2].bar(xx, [100*R[c]['intervals']['RCHE-ACI (80%)']['coverage'] for c, _ in cases], color='#9a62d6'); axs[2].axhline(80, color='#d0453a', ls='--', lw=2)
    axs[2].set_ylabel('ACI cover. (%)'); axs[2].set_ylim(60, 100); axs[2].set_xticks(xx, [c for c, _ in cases], rotation=30, ha='right')
    fig.tight_layout(); fig.savefig('figs/fig9_sensitivity.png'); plt.close(fig)
print('ok')
