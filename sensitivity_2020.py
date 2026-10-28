"""Sensitivity of the main results to the IMF October 2026 revision of 2020 inflation (4.5% -> -2.7%).
Copies the repository to a temporary folder, replaces the 2020 value, re-runs model.py and evaluate.py
and prints the main metrics next to the published ones. Run from the repository root: python3 sensitivity_2020.py"""
import json, os, shutil, subprocess, sys, tempfile
import pandas as pd
root = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp()
shutil.copytree(os.path.join(root, 'data'), os.path.join(tmp, 'data'))
shutil.copytree(os.path.join(root, 'work'), os.path.join(tmp, 'work'), ignore=shutil.ignore_patterns('figs'))
d = pd.read_csv(os.path.join(tmp, 'data', 'chad.csv'))
d.loc[d.year == 2020, 'infl'] = -2.7
d.loc[d.year == 2020, 'misery'] = -2.7 + d.loc[d.year == 2020, 'unemp']
d.to_csv(os.path.join(tmp, 'data', 'chad.csv'), index=False)
w = os.path.join(tmp, 'work')
for cmd in (['model.py'], ['evaluate.py', 'base', '2005']):
    subprocess.run([sys.executable] + cmd, cwd=w, check=True, stdout=subprocess.DEVNULL)
def metrics(e):
    return {'Median MAE': e['point']['MED']['MAE'], 'RCHE MAE': e['point']['ens']['MAE'],
            'DM p, RCHE vs median': e['dm_vs_MED']['ens']['p_abs'], 'DM p, RCHE vs random walk': e['dm_vs_RW']['ens']['p_abs'],
            'ACI coverage (80%)': e['intervals']['RCHE-ACI (80%)']['coverage'], 'ACI width': e['intervals']['RCHE-ACI (80%)']['width'],
            'CRPS RCHE': e['crps']['RCHE'], 'CRPS climatology': e['crps']['Climatology']}
a = metrics(json.load(open(os.path.join(root, 'work', 'eval_base_2005.json'))))
b = metrics(json.load(open(os.path.join(w, 'eval_base_2005.json'))))
print(f"{'metric':28s} {'published':>10s} {'2020 revised':>13s}")
for k in a: print(f"{k:28s} {a[k]:10.3f} {b[k]:13.3f}")
shutil.rmtree(tmp)
