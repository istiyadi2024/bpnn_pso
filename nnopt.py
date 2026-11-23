"""
nnopt.py - Neural network (1 hidden layer) trained by:
  * Backpropagation (mini-batch SGD + momentum, early stopping)   -> BPNN
  * Particle Swarm Optimization (gbest PSO)                        -> PSO-NN
  * Real-coded Genetic Algorithm                                   -> GA-NN
  * Two-stage hybrid: PSO global search + BP fine-tuning           -> PSO-BP

Rewritten from the thesis package `hybridnn` with these fixes:
  - pbest / gbest are stored as COPIES (the original code stored references,
    so pbest/gbest drifted with the particle and the "best" weights were lost);
  - bias terms are used;
  - linear output layer (a regression target can be extrapolated);
  - PSO acceleration coefficients c1, c2 are independent of the BP learning rate;
  - model selection uses a validation set only (the test set is never touched).

Only depends on numpy.
"""
import time
import numpy as np


# ---------------------------------------------------------------- network ---
class MLPShape:
    """One hidden layer (tanh) + linear output. Parameters are a flat vector."""

    def __init__(self, n_in, n_hidden):
        self.n_in, self.h = n_in, n_hidden
        self.dim = n_in * n_hidden + n_hidden + n_hidden + 1

    def unpack(self, theta):
        """theta: (D,) or (P, D) -> W1 (P,n_in,h), b1 (P,h), W2 (P,h), b2 (P,)"""
        t = np.atleast_2d(theta)
        i, h = self.n_in, self.h
        a = i * h
        W1 = t[:, :a].reshape(-1, i, h)
        b1 = t[:, a:a + h]
        W2 = t[:, a + h:a + 2 * h]
        b2 = t[:, a + 2 * h]
        return W1, b1, W2, b2

    def predict_pop(self, theta, X):
        """Predictions for a whole population: returns (P, N)."""
        W1, b1, W2, b2 = self.unpack(theta)
        H = np.tanh(np.einsum('ni,pih->pnh', X, W1) + b1[:, None, :])
        return np.einsum('pnh,ph->pn', H, W2) + b2[:, None]

    def predict(self, theta, X):
        return self.predict_pop(theta, X)[0]

    def rmse_pop(self, theta, X, y):
        e = self.predict_pop(theta, X) - y[None, :]
        return np.sqrt(np.mean(e * e, axis=1))

    def init(self, rng, size=None):
        """Xavier/Glorot uniform initialisation (one vector or a population)."""
        n = 1 if size is None else size
        lim1 = np.sqrt(6.0 / (self.n_in + self.h))
        lim2 = np.sqrt(6.0 / (self.h + 1))
        a = self.n_in * self.h
        th = np.empty((n, self.dim))
        th[:, :a] = rng.uniform(-lim1, lim1, (n, a))
        th[:, a:a + self.h] = 0.0
        th[:, a + self.h:a + 2 * self.h] = rng.uniform(-lim2, lim2, (n, self.h))
        th[:, -1] = 0.0
        return th[0] if size is None else th


# ------------------------------------------------------------------ BPNN ----
def train_bp(net, X, y, Xv, yv, rng, lr=0.05, momentum=0.9, batch=8,
             max_epochs=3000, patience=200, theta0=None):
    """Mini-batch SGD with momentum; early stopping on validation RMSE.
    Returns dict(theta, curve_train, curve_val, epochs, time)."""
    t0 = time.perf_counter()
    theta = net.init(rng) if theta0 is None else theta0.copy()
    vel = np.zeros_like(theta)
    best, best_val, wait = theta.copy(), np.inf, 0
    ct, cv = [], []
    n, i, h = len(X), net.n_in, net.h
    a = i * h
    for ep in range(max_epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            xb, yb = X[b], y[b]
            W1 = theta[:a].reshape(i, h); b1 = theta[a:a + h]
            W2 = theta[a + h:a + 2 * h]; b2 = theta[-1]
            Z = xb @ W1 + b1
            Hh = np.tanh(Z)
            out = Hh @ W2 + b2
            d = (out - yb) / len(b)                 # dMSE/dout (x0.5 factor folded)
            gW2 = Hh.T @ d
            gb2 = d.sum()
            dH = np.outer(d, W2) * (1 - Hh * Hh)
            gW1 = xb.T @ dH
            gb1 = dH.sum(axis=0)
            g = np.concatenate([gW1.ravel(), gb1, gW2, [gb2]])
            vel = momentum * vel - lr * g
            theta = theta + vel
        tr = net.rmse_pop(theta, X, y)[0]
        va = net.rmse_pop(theta, Xv, yv)[0]
        ct.append(tr); cv.append(va)
        if not np.isfinite(tr):
            break
        if va < best_val - 1e-7:
            best_val, best, wait = va, theta.copy(), 0
        else:
            wait += 1
            if wait >= patience:
                break
    return dict(theta=best, curve_train=np.array(ct), curve_val=np.array(cv),
                epochs=ep + 1, time=time.perf_counter() - t0, val_rmse=best_val)


# ------------------------------------------------------------------- PSO ----
def train_pso(net, X, y, Xv, yv, rng, n_particles=50, iters=300,
              w=0.7298, w_end=None, c1=1.49445, c2=1.49445,
              bound=5.0, vmax_frac=0.2):
    """gbest PSO minimising training RMSE.
    w_end=None -> constant inertia w; otherwise linear decrease w -> w_end.
    The returned solution is the gbest snapshot with the lowest validation RMSE
    (validation-based model selection; the test set is never used)."""
    t0 = time.perf_counter()
    D = net.dim
    vmax = vmax_frac * 2 * bound
    pos = net.init(rng, n_particles)
    vel = rng.uniform(-0.1, 0.1, (n_particles, D))
    fit = net.rmse_pop(pos, X, y)
    pbest, pbest_fit = pos.copy(), fit.copy()             # COPIES (bug fix)
    g = int(np.argmin(pbest_fit))
    gbest, gbest_fit = pbest[g].copy(), pbest_fit[g]
    sel, sel_val = gbest.copy(), net.rmse_pop(gbest, Xv, yv)[0]
    ct, cv = [gbest_fit], [sel_val]
    for t in range(iters):
        wt = w if w_end is None else w - (w - w_end) * t / max(iters - 1, 1)
        r1 = rng.random((n_particles, D)); r2 = rng.random((n_particles, D))
        vel = wt * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        np.clip(vel, -vmax, vmax, out=vel)
        pos = np.clip(pos + vel, -bound, bound)
        fit = net.rmse_pop(pos, X, y)
        imp = fit < pbest_fit
        pbest[imp] = pos[imp]
        pbest_fit[imp] = fit[imp]
        g = int(np.argmin(pbest_fit))
        if pbest_fit[g] < gbest_fit:
            gbest, gbest_fit = pbest[g].copy(), pbest_fit[g]
            v = net.rmse_pop(gbest, Xv, yv)[0]
            if v < sel_val:
                sel, sel_val = gbest.copy(), v
        ct.append(gbest_fit); cv.append(sel_val)
    return dict(theta=sel, curve_train=np.array(ct), curve_val=np.array(cv),
                nfe=n_particles * (iters + 1), time=time.perf_counter() - t0,
                val_rmse=sel_val)


# -------------------------------------------------------------------- GA ----
def train_ga(net, X, y, Xv, yv, rng, pop=50, gens=300, cx_rate=0.9,
             mut_rate=None, mut_sigma=0.3, elite=2, tour=3, bound=5.0, alpha=0.5):
    """Real-coded GA: tournament selection, BLX-alpha crossover,
    Gaussian mutation (sigma decays linearly), elitism."""
    t0 = time.perf_counter()
    D = net.dim
    mut_rate = 1.0 / D if mut_rate is None else mut_rate
    P = net.init(rng, pop)
    fit = net.rmse_pop(P, X, y)
    b = int(np.argmin(fit))
    best, best_fit = P[b].copy(), fit[b]
    sel, sel_val = best.copy(), net.rmse_pop(best, Xv, yv)[0]
    ct, cv = [best_fit], [sel_val]
    for gnr in range(gens):
        sigma = mut_sigma * (1 - gnr / gens) + 0.01
        order = np.argsort(fit)
        children = [P[order[k]].copy() for k in range(elite)]
        while len(children) < pop:
            c = rng.integers(0, pop, (2, tour))
            p1 = P[c[0][np.argmin(fit[c[0]])]]
            p2 = P[c[1][np.argmin(fit[c[1]])]]
            if rng.random() < cx_rate:
                lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
                span = hi - lo
                ch = rng.uniform(lo - alpha * span, hi + alpha * span)
            else:
                ch = p1.copy()
            m = rng.random(D) < mut_rate
            ch[m] += rng.normal(0, sigma, m.sum())
            children.append(np.clip(ch, -bound, bound))
        P = np.array(children)
        fit = net.rmse_pop(P, X, y)
        b = int(np.argmin(fit))
        if fit[b] < best_fit:
            best, best_fit = P[b].copy(), fit[b]
            v = net.rmse_pop(best, Xv, yv)[0]
            if v < sel_val:
                sel, sel_val = best.copy(), v
        ct.append(best_fit); cv.append(sel_val)
    return dict(theta=sel, curve_train=np.array(ct), curve_val=np.array(cv),
                nfe=pop * (gens + 1), time=time.perf_counter() - t0, val_rmse=sel_val)


# --------------------------------------------------------------- PSO-BP -----
def train_pso_bp(net, X, y, Xv, yv, rng, pso_kw=None, bp_kw=None):
    """Stage 1: PSO explores the weight space (global search).
    Stage 2: BP fine-tunes the PSO solution (local search), early stopping."""
    t0 = time.perf_counter()
    r1 = train_pso(net, X, y, Xv, yv, rng, **(pso_kw or {}))
    r2 = train_bp(net, X, y, Xv, yv, rng, theta0=r1['theta'], **(bp_kw or {}))
    theta, val = (r2['theta'], r2['val_rmse']) if r2['val_rmse'] <= r1['val_rmse'] \
        else (r1['theta'], r1['val_rmse'])
    return dict(theta=theta, curve_train=np.concatenate([r1['curve_train'], r2['curve_train']]),
                stage1_len=len(r1['curve_train']), time=time.perf_counter() - t0,
                val_rmse=val, epochs=r2['epochs'])
