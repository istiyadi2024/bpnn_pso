"""make_figures.py - journal figures (PNG, 300 dpi) from ./results"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_experiment as R

OUT = R.OUT
INK, INK2, GRID = '#222222', '#666666', '#e6e6e6'
C = {'BPNN': '#2a78d6', 'PSO-NN': '#eb6834', 'GA-NN': '#1baf7a', 'PSO-BP': '#eda100'}
NN = ['BPNN', 'PSO-NN', 'GA-NN', 'PSO-BP']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'legend.frameon': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig)


P = R.prepare(); d = P['d']
runs = pd.read_csv(os.path.join(OUT, 'runs_final.csv'))
base = pd.read_csv(os.path.join(OUT, 'summary_baselines.csv'))
med = np.load(os.path.join(OUT, 'preds_median.npz'))

# 1. data and split ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 2.6))
ax.plot(P['df'].date, P['df'].emas, color=INK, lw=1.5)
for sl, lab, a in [(P['tr'], 'Training', 0.0), (P['va'], 'Validation', 0.10), (P['te'], 'Test', 0.20)]:
    x0, x1 = d.date.iloc[sl.start], d.date.iloc[sl.stop - 1]
    if a:
        ax.axvspan(x0, x1, color='#2a78d6', alpha=a, lw=0)
    ax.text(x0 + (x1 - x0) / 2, ax.get_ylim()[1] if False else 2000, lab, ha='center', color=INK2, fontsize=8)
ax.set_ylabel('Gold price (USD/oz)'); ax.set_ylim(1000, 2080)
save(fig, 'fig1_data_split.png')

# 2. test RMSE distribution -------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.0))
data = [runs[runs.model == m].RMSE.values for m in NN]
bp = ax.boxplot(data, tick_labels=NN, widths=0.5, patch_artist=True, showfliers=True,
                medianprops=dict(color=INK, lw=1.5), flierprops=dict(marker='o', ms=3, mec=INK2))
for patch, m in zip(bp['boxes'], NN):
    patch.set_facecolor(C[m]); patch.set_alpha(0.85); patch.set_edgecolor(INK2)
rw = base.loc[base.model.str.startswith('Naive'), 'RMSE'].iloc[0]
ax.axhline(rw, color=INK, ls='--', lw=1)
ax.text(4.45, rw, f' Naive / ARIMA\n {rw:.1f}', va='center', fontsize=7.5, color=INK)
ax.set_ylabel('Test RMSE (USD/oz), 30 runs')
save(fig, 'fig2_test_rmse_boxplot.png')

# 3. forecasts on the test set (median runs) --------------------------------
dates = d.date.iloc[P['te']]
fig, ax = plt.subplots(figsize=(6.5, 3.0))
ax.plot(dates, med['y_test'], color=INK, lw=2, label='Actual')
ax.plot(dates, med['Naive'], color=INK2, lw=1.2, ls='--', label='Naive')
for m in ['BPNN', 'PSO-NN', 'PSO-BP']:
    ax.plot(dates, med[m], color=C[m], lw=1.5, marker='o', ms=3, label=f'{m} (median run)')
ax.set_ylabel('Gold price (USD/oz)'); ax.legend(ncol=5, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.22))
fig.autofmt_xdate()
save(fig, 'fig3_test_forecasts.png')

# 4. convergence PSO vs GA --------------------------------------------------
cv = np.load(os.path.join(OUT, 'curves_final.npz'))
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for m in ['PSO-NN', 'GA-NN']:
    c = cv[m]; it = np.arange(c.shape[1])
    ax.plot(it, np.median(c, 0), color=C[m], lw=2, label=m)
    ax.fill_between(it, np.percentile(c, 25, 0), np.percentile(c, 75, 0), color=C[m], alpha=0.2, lw=0)
ax.set_xlabel('Iteration / generation'); ax.set_ylabel('Training RMSE (scaled)')
ax.legend()
save(fig, 'fig4_convergence.png')

# 5. PSO sensitivity heatmap ------------------------------------------------
s = pd.read_csv(os.path.join(OUT, 'sensitivity_pso.csv'))
s = s[s.c1_c2 > 1]
order = ['0.4', '0.6', '0.7298', 'LDIW 0.9->0.4']
s['inertia'] = s.inertia.astype(str)
piv = s.pivot(index='swarm', columns='inertia', values='val_rmse_mean')[order]
fig, ax = plt.subplots(figsize=(4.8, 2.8))
im = ax.imshow(piv.values, cmap='Blues', aspect='auto')
ax.grid(False)
ax.set_xticks(range(len(order)), ['0.4', '0.6', '0.7298', 'LDIW\n0.9→0.4'])
ax.set_yticks(range(len(piv.index)), piv.index)
ax.set_xlabel('Inertia weight w'); ax.set_ylabel('Swarm size')
vmid = np.nanmean(piv.values)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8,
                color='white' if v > vmid else INK)
fig.colorbar(im, ax=ax, label='Validation RMSE (scaled)')
save(fig, 'fig5_pso_sensitivity.png')

# 6. protocol check ---------------------------------------------------------
if not os.path.exists(os.path.join(OUT, 'protocol_check_runs.csv')):
    print('figures saved'); raise SystemExit
pc = pd.read_csv(os.path.join(OUT, 'protocol_check_runs.csv'))
ms = [m for m in ['BPNN (original hybridnn)', 'NNPSO (original hybridnn)', 'BPNN (fixed)', 'PSO-NN (fixed)']
      if m in pc.model.unique()]
fig, ax = plt.subplots(figsize=(6.0, 3.0))
bpx = ax.boxplot([pc[pc.model == m].RMSE.values for m in ms], widths=0.5, patch_artist=True,
                 tick_labels=[m.replace(' (original hybridnn)', '\noriginal').replace(' (fixed)', '\nfixed') for m in ms],
                 medianprops=dict(color=INK, lw=1.5), flierprops=dict(marker='o', ms=3, mec=INK2))
cols = ['#2a78d6', '#eb6834', '#2a78d6', '#eb6834']
for p_, c_ in zip(bpx['boxes'], cols):
    p_.set_facecolor(c_); p_.set_alpha(0.85); p_.set_edgecolor(INK2)
ax.set_ylabel('Test RMSE (USD/oz), 30 runs')
ax.set_title('Thesis protocol (random split, scaler on all data)', fontsize=9, color=INK)
save(fig, 'fig6_protocol_check.png')
print('figures saved')
