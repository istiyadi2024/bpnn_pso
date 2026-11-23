"""
Protocol (fixes for journal review):
  1. One-step-ahead FORECASTING: inputs are lagged values only
     (gold t-1..t-3, oil t-1, USD/IDR t-1) -> target gold t.
  2. Chronological split: train 70% | validation 15% | test 15%.
  3. Min-max scaler fitted on the TRAINING set only.
  4. All hyper-parameters chosen on the VALIDATION set; test used once.
  5. 30 independent runs (seeds 1..30) per stochastic model.
  6. Metrics in USD/oz: RMSE, MAE, MAPE, R2.
  7. Baselines: Naive (random walk), ARIMA, multiple linear regression.
  8. Statistics: Friedman + Wilcoxon signed-rank (Holm), Diebold-Mariano (HLN).
  9. Extra: protocol check showing the effect of the thesis protocol
     (random split + scaler on all data) and of the original hybridnn PSO bug.

Usage:  python run_experiment.py            (all stages, ~10-20 min)
Outputs go to ./results
"""
import os, sys, json, time, itertools, warnings
import numpy as np
import pandas as pd
from scipy import stats

from nnopt import MLPShape, train_bp, train_pso, train_ga, train_pso_bp

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('TARGET', 'level')      # 'level' or 'return'
OUT = os.path.join(HERE, 'results' if MODE == 'level' else 'results_return')
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(HERE, '..', 'dataset.csv')

N_SEEDS_TUNE = 10
N_SEEDS_FINAL = 30
LAGS_GOLD = 3


# ------------------------------------------------------------------ data ----
def load_data(path=DATA):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(dict(year=df.tahun, month=df.bulan, day=1))
    df = df.sort_values('date').reset_index(drop=True)
    return df


def make_supervised(df):
    if MODE == 'return':
        # stationary target: log-return of gold; inputs: lagged log-returns
        r = np.log(df[['emas', 'minyak', 'dolar']]).diff()
        d = pd.DataFrame({'date': df['date'], 'y': r['emas'], 'prev_price': df['emas'].shift(1),
                          'price': df['emas']})
        for k in range(1, LAGS_GOLD + 1):
            d[f'gold_ret_lag{k}'] = r['emas'].shift(k)
        d['oil_ret_lag1'] = r['minyak'].shift(1)
        d['usd_ret_lag1'] = r['dolar'].shift(1)
        return d.dropna().reset_index(drop=True)
    d = pd.DataFrame({'date': df['date'], 'y': df['emas']})
    for k in range(1, LAGS_GOLD + 1):
        d[f'gold_lag{k}'] = df['emas'].shift(k)
    d['oil_lag1'] = df['minyak'].shift(1)
    d['usd_lag1'] = df['dolar'].shift(1)
    return d.dropna().reset_index(drop=True)


class Scaler:
    def fit(self, a):
        self.lo, self.hi = a.min(0), a.max(0); return self
    def transform(self, a):
        return (a - self.lo) / (self.hi - self.lo)
    def inverse(self, a):
        return a * (self.hi - self.lo) + self.lo


def chrono_split(d, val_frac=0.15, test_frac=0.15):
    n = len(d)
    n_test = int(round(n * test_frac)); n_val = int(round(n * val_frac))
    n_tr = n - n_val - n_test
    return slice(0, n_tr), slice(n_tr, n_tr + n_val), slice(n_tr + n_val, n)


def prepare():
    df = load_data()
    d = make_supervised(df)
    feats = [c for c in d.columns if c not in ('date', 'y', 'prev_price', 'price')]
    tr, va, te = chrono_split(d)
    X, y = d[feats].values.astype(float), d['y'].values.astype(float)
    sx = Scaler().fit(X[tr]); sy = Scaler().fit(y[tr])
    Xs, ys = sx.transform(X), sy.transform(y)
    if MODE == 'return':
        price, prev = d['price'].values, d['prev_price'].values
        to_price = lambda pred_scaled, sl: prev[sl] * np.exp(sy.inverse(pred_scaled))
    else:
        price = y
        to_price = lambda pred_scaled, sl: sy.inverse(pred_scaled)
    return dict(df=df, d=d, feats=feats, tr=tr, va=va, te=te, X=X, y=y, price=price,
                Xs=Xs, ys=ys, sx=sx, sy=sy, to_price=to_price)


def metrics(y, p):
    e = y - p
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                MAPE=float(np.mean(np.abs(e / y)) * 100),
                R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)))


# -------------------------------------------------------------- baselines ---
def baselines(P):
    d, te, tr = P['d'], P['te'], P['tr']
    y_te = P['price'][te]
    out, preds = [], {}
    # Naive / random walk
    p = (d['prev_price'] if MODE == 'return' else d['gold_lag1']).values[te]
    out.append(dict(model='Naive (random walk)', **metrics(y_te, p))); preds['Naive'] = p
    # Multiple linear regression (OLS) on the same lagged features, train only
    Xtr = np.c_[np.ones(tr.stop), P['X'][tr]]
    beta, *_ = np.linalg.lstsq(Xtr, P['y'][tr], rcond=None)
    p = np.c_[np.ones(len(y_te)), P['X'][te]] @ beta
    if MODE == 'return':
        p = d['prev_price'].values[te] * np.exp(p)
    out.append(dict(model='MLR (OLS)', **metrics(y_te, p))); preds['MLR'] = p
    # ARIMA: order chosen by AIC on the training part, one-step-ahead on test
    from statsmodels.tsa.arima.model import ARIMA
    gold = P['df'].set_index('date')['emas'].asfreq('MS')
    offset = len(P['df']) - len(d)              # rows lost to lags
    n_train_raw = offset + tr.stop
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for p_, q_ in itertools.product(range(4), range(4)):
            try:
                r = ARIMA(gold.iloc[:n_train_raw], order=(p_, 1, q_), trend='t').fit()
                if best is None or r.aic < best[0]:
                    best = (r.aic, (p_, 1, q_), r)
            except Exception:
                pass
        res_full = best[2].apply(gold)          # same parameters, full series
        pr = res_full.get_prediction(start=offset + te.start, end=len(gold) - 1, dynamic=False)
        p = pr.predicted_mean.values
    out.append(dict(model=f'ARIMA{best[1]}', **metrics(y_te, p))); preds['ARIMA'] = p
    return pd.DataFrame(out), preds


# --------------------------------------------------------- tuning stages ----
def data_arrays(P):
    Xs, ys = P['Xs'], P['ys']
    return Xs[P['tr']], ys[P['tr']], Xs[P['va']], ys[P['va']]


def tune_bpnn(P):
    X, y, Xv, yv = data_arrays(P)
    rows = []
    for H, lr in itertools.product([3, 5, 7, 9], [0.01, 0.05, 0.1]):
        net = MLPShape(X.shape[1], H)
        v = [train_bp(net, X, y, Xv, yv, np.random.default_rng(s), lr=lr)['val_rmse']
             for s in range(101, 101 + N_SEEDS_TUNE)]
        rows.append(dict(hidden=H, lr=lr, val_rmse_mean=np.mean(v), val_rmse_std=np.std(v)))
        print(f'  BPNN H={H} lr={lr}: val {np.mean(v):.4f}', flush=True)
    t = pd.DataFrame(rows).sort_values('val_rmse_mean')
    t.to_csv(os.path.join(OUT, 'tuning_bpnn.csv'), index=False)
    return int(t.iloc[0].hidden), float(t.iloc[0].lr)


def tune_pso(P, H):
    """Sensitivity of PSO-NN to swarm size and inertia weight (validation)."""
    X, y, Xv, yv = data_arrays(P)
    net = MLPShape(X.shape[1], H)
    configs = []
    for n, wl in itertools.product([10, 30, 50, 100], ['0.4', '0.6', '0.7298', 'LDIW 0.9->0.4']):
        w, w_end = (0.9, 0.4) if wl.startswith('LDIW') else (float(wl), None)
        configs.append(dict(swarm=n, inertia=wl, c=1.49445, w=w, w_end=w_end))
    for wl in ['0.4', '0.5', '0.6']:   # thesis-style: c1 = c2 = learning rate 0.7
        configs.append(dict(swarm=50, inertia=wl, c=0.7, w=float(wl), w_end=None))
    rows = []
    for c in configs:
        v, tm = [], []
        for s in range(101, 101 + N_SEEDS_TUNE):
            r = train_pso(net, X, y, Xv, yv, np.random.default_rng(s), n_particles=c['swarm'],
                          iters=300, w=c['w'], w_end=c['w_end'], c1=c['c'], c2=c['c'])
            v.append(r['val_rmse']); tm.append(r['time'])
        rows.append(dict(swarm=c['swarm'], inertia=c['inertia'], c1_c2=c['c'],
                         val_rmse_mean=np.mean(v), val_rmse_std=np.std(v), time_s=np.mean(tm),
                         w=c['w'], w_end=c['w_end']))
        print(f"  PSO n={c['swarm']} w={c['inertia']} c={c['c']}: val {np.mean(v):.4f}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, 'sensitivity_pso.csv'), index=False)
    best = t[t.c1_c2 > 1].sort_values('val_rmse_mean').iloc[0]
    return dict(n_particles=int(best.swarm), w=float(best.w),
                w_end=None if pd.isna(best.w_end) else float(best.w_end))


def tune_ga(P, H):
    X, y, Xv, yv = data_arrays(P)
    net = MLPShape(X.shape[1], H)
    rows = []
    for sig, mr in itertools.product([0.1, 0.3], [None, 0.1]):
        v = [train_ga(net, X, y, Xv, yv, np.random.default_rng(s), mut_sigma=sig,
                      mut_rate=mr)['val_rmse'] for s in range(101, 101 + N_SEEDS_TUNE)]
        rows.append(dict(mut_sigma=sig, mut_rate='1/D' if mr is None else mr,
                         val_rmse_mean=np.mean(v), val_rmse_std=np.std(v)))
        print(f'  GA sigma={sig} mr={mr}: val {np.mean(v):.4f}', flush=True)
    t = pd.DataFrame(rows).sort_values('val_rmse_mean')
    t.to_csv(os.path.join(OUT, 'tuning_ga.csv'), index=False)
    b = t.iloc[0]
    return dict(mut_sigma=float(b.mut_sigma), mut_rate=None if b.mut_rate == '1/D' else float(b.mut_rate))


# ------------------------------------------------------------ final runs ----
def final_runs(P, H, lr, pso_cfg, ga_cfg):
    X, y, Xv, yv = data_arrays(P)
    Xte, y_te = P['Xs'][P['te']], P['price'][P['te']]
    net = MLPShape(X.shape[1], H)
    models = {
        'BPNN': lambda rng: train_bp(net, X, y, Xv, yv, rng, lr=lr),
        'PSO-NN': lambda rng: train_pso(net, X, y, Xv, yv, rng, iters=300, **pso_cfg),
        'GA-NN': lambda rng: train_ga(net, X, y, Xv, yv, rng, pop=50, gens=300, **ga_cfg),
        'PSO-BP': lambda rng: train_pso_bp(net, X, y, Xv, yv, rng,
                                           pso_kw=dict(iters=150, **pso_cfg), bp_kw=dict(lr=lr)),
    }
    rows, preds, curves = [], {m: [] for m in models}, {m: [] for m in models}
    for m, fn in models.items():
        for s in range(1, N_SEEDS_FINAL + 1):
            r = fn(np.random.default_rng(s))
            p = P['to_price'](net.predict(r['theta'], Xte), P['te'])
            rows.append(dict(model=m, seed=s, time_s=r['time'],
                             val_rmse_scaled=r['val_rmse'], **metrics(y_te, p)))
            preds[m].append(p); curves[m].append(r['curve_train'])
        print(f'  {m}: done', flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(OUT, 'runs_final.csv'), index=False)
    np.savez(os.path.join(OUT, 'preds_final.npz'), **{m: np.array(v) for m, v in preds.items()})
    # convergence curves (pad with last value)
    cv = {}
    for m, lst in curves.items():
        L = max(len(c) for c in lst)
        cv[m] = np.array([np.pad(c, (0, L - len(c)), mode='edge') for c in lst])
    np.savez(os.path.join(OUT, 'curves_final.npz'), **cv)
    return runs, preds


# ------------------------------------------------------------ statistics ----
def dm_test(e1, e2, h=1):
    """Diebold-Mariano test with Harvey-Leybourne-Newbold correction, squared loss."""
    d = e1 ** 2 - e2 ** 2
    n = len(d); dbar = d.mean()
    gam = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    dm = dbar / np.sqrt(var)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return float(hln), float(p)


def holm(pvals):
    p = np.asarray(pvals); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(run, 1.0)
    return adj


def statistics(P, runs, preds, base_preds):
    y_te = P['price'][P['te']]
    nn = ['BPNN', 'PSO-NN', 'GA-NN', 'PSO-BP']
    piv = runs.pivot(index='seed', columns='model', values='RMSE')[nn]
    fr = stats.friedmanchisquare(*[piv[m].values for m in nn])
    pairs = [('PSO-NN', 'BPNN'), ('GA-NN', 'BPNN'), ('PSO-BP', 'BPNN'),
             ('PSO-BP', 'PSO-NN'), ('PSO-NN', 'GA-NN'), ('PSO-BP', 'GA-NN')]
    w = [stats.wilcoxon(piv[a], piv[b]) for a, b in pairs]
    adj = holm([x.pvalue for x in w])
    wil = pd.DataFrame([dict(model_A=a, model_B=b, median_RMSE_A=piv[a].median(),
                             median_RMSE_B=piv[b].median(), W=x.statistic, p=x.pvalue, p_holm=pa,
                             A_better_in_runs=int((piv[a] < piv[b]).sum()))
                        for (a, b), x, pa in zip(pairs, w, adj)])
    wil.to_csv(os.path.join(OUT, 'stats_wilcoxon.csv'), index=False)
    # DM: median-RMSE run of each NN model vs naive and vs BPNN median run
    med = {}
    for m in nn:
        r = runs[runs.model == m].reset_index(drop=True)
        i = int((r.RMSE - r.RMSE.median()).abs().idxmin())
        med[m] = preds[m][i]
    allp = {**{k: v for k, v in base_preds.items()}, **med}
    rows = []
    for m in ['MLR', 'ARIMA'] + nn:
        for ref in ['Naive', 'BPNN']:
            if m == ref:
                continue
            s, p = dm_test(y_te - allp[m], y_te - allp[ref])
            rows.append(dict(model=m, reference=ref, DM_HLN=s, p=p))
    dm = pd.DataFrame(rows)
    dm.to_csv(os.path.join(OUT, 'stats_dm.csv'), index=False)
    np.savez(os.path.join(OUT, 'preds_median.npz'), y_test=y_te, **allp)
    return dict(friedman_chi2=float(fr.statistic), friedman_p=float(fr.pvalue)), wil, dm


# -------------------------------------------------------- protocol check ----
def protocol_check(H, lr, pso_cfg):
    """Same fixed models under the THESIS protocol: features (month, year, oil,
    USD/IDR) of the same month, scaler fitted on ALL data, random 80/20 split
    (seed 12). Plus the original buggy hybridnn NNPSO with the thesis best config."""
    df = load_data()
    X = df[['bulan', 'tahun', 'minyak', 'dolar']].values.astype(float)
    y = df['emas'].values.astype(float)
    sx, sy = Scaler().fit(X), Scaler().fit(y)          # leakage: all data
    Xs, ys = sx.transform(X), sy.transform(y)
    idx = np.random.RandomState(12).permutation(len(X))
    nte = int(len(X) * 0.2); te, tr = idx[:nte], idx[nte:]
    rows = []
    net = MLPShape(4, H)
    for s in range(1, N_SEEDS_FINAL + 1):
        # no validation set in the thesis protocol -> use test for early stopping,
        # exactly the optimistic practice we want to quantify
        rb = train_bp(net, Xs[tr], ys[tr], Xs[te], ys[te], np.random.default_rng(s), lr=lr)
        rp = train_pso(net, Xs[tr], ys[tr], Xs[te], ys[te], np.random.default_rng(s), iters=300, **pso_cfg)
        for m, r in [('BPNN (fixed)', rb), ('PSO-NN (fixed)', rp)]:
            p = sy.inverse(net.predict(r['theta'], Xs[te]))
            rows.append(dict(protocol='thesis (random split, leakage)', model=m, seed=s, **metrics(y[te], p)))
    # original package, thesis best configuration
    sys.path.insert(0, os.path.join(HERE, '..', 'hybridnn-main', 'src'))
    try:
        import hybridnn as hb
        Xo = (X - X.min(0)) / (X.max(0) - X.min(0)); yo = ((y - y.min()) / (y.max() - y.min())).reshape(-1, 1)
        for s in range(1, N_SEEDS_FINAL + 1):
            np.random.seed(1000 + s)
            b = hb.Backpropagation(4, [7], [hb.Tanh(), hb.Sigmoid()], 1, 0.7)
            b.train(Xo[tr], yo[tr], 30)
            n = hb.NNPSO(4, [7], [hb.Tanh(), hb.Sigmoid()], 1, 0.7, 50, 30, 0.6)
            n.initialize(); n.optimize(Xo[tr], yo[tr])
            for m, obj in [('BPNN (original hybridnn)', b), ('NNPSO (original hybridnn)', n)]:
                p = obj.predict(Xo[te]).ravel() * (y.max() - y.min()) + y.min()
                rows.append(dict(protocol='thesis (random split, leakage)', model=m, seed=s, **metrics(y[te], p)))
    except Exception as e:
        print('  original hybridnn not run:', e)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, 'protocol_check_runs.csv'), index=False)
    return t


# ------------------------------------------------------------------ main ----
def main():
    t0 = time.time()
    P = prepare()
    d = P['d']
    info = dict(n_samples=len(d), features=P['feats'],
                train=[str(d.date[P['tr']].iloc[0].date()), str(d.date[P['tr']].iloc[-1].date()), P['tr'].stop],
                val=[str(d.date[P['va']].iloc[0].date()), str(d.date[P['va']].iloc[-1].date()), P['va'].stop - P['va'].start],
                test=[str(d.date[P['te']].iloc[0].date()), str(d.date[P['te']].iloc[-1].date()), P['te'].stop - P['te'].start])
    print(json.dumps(info, indent=1), flush=True)
    print('[1] baselines'); base, base_preds = baselines(P)
    print('[2] tune BPNN'); H, lr = tune_bpnn(P)
    print('[3] tune PSO'); pso_cfg = tune_pso(P, H)
    print('[4] tune GA'); ga_cfg = tune_ga(P, H)
    info.update(hidden=H, lr=lr, pso=pso_cfg, ga=ga_cfg)
    print('    chosen:', info)
    print('[5] final 30 runs'); runs, preds = final_runs(P, H, lr, pso_cfg, ga_cfg)
    print('[6] statistics'); fr, wil, dm = statistics(P, runs, preds, base_preds)
    info.update(fr)
    summ = runs.groupby('model')[['RMSE', 'MAE', 'MAPE', 'R2', 'time_s']].agg(['mean', 'std', 'min', 'median'])
    summ.columns = ['_'.join(c) for c in summ.columns]
    summ = summ.reindex(['BPNN', 'PSO-NN', 'GA-NN', 'PSO-BP'])
    summ.to_csv(os.path.join(OUT, 'summary_nn.csv'))
    base.to_csv(os.path.join(OUT, 'summary_baselines.csv'), index=False)
    if MODE != 'level':
        info['runtime_min'] = (time.time() - t0) / 60
        with open(os.path.join(OUT, 'config_used.json'), 'w') as f:
            json.dump(info, f, indent=1, default=str)
        print(base.round(3).to_string()); print(summ.round(3).to_string())
        print(wil.round(4).to_string()); print(dm.round(4).to_string())
        return
    print('[7] protocol check'); pc = protocol_check(H, lr, pso_cfg)
    pcs = pc.groupby('model')[['RMSE', 'MAPE']].agg(['mean', 'std', 'min', 'max'])
    pcs.columns = ['_'.join(c) for c in pcs.columns]
    pcs.to_csv(os.path.join(OUT, 'protocol_check_summary.csv'))
    info['runtime_min'] = (time.time() - t0) / 60
    with open(os.path.join(OUT, 'config_used.json'), 'w') as f:
        json.dump(info, f, indent=1, default=str)
    print(base.round(3).to_string()); print(summ.round(3).to_string())
    print(wil.round(4).to_string()); print(dm.round(4).to_string()); print(pcs.round(3).to_string())
    print('done in %.1f min' % info['runtime_min'])


if __name__ == '__main__':
    main()
