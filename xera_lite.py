"""xERA-lite: a fitted map from a pitcher's leakage-safe as-of Savant/warehouse
features to expected runs allowed per 9 innings (runs/9).

This is the INPUT to mlb_starters.expected_runs_additive — the Savant-pitcher
design determination's resolution of the fit<->serve<->grade scale trap. Instead
of dividing two different-scale xwOBAs (contact xwOBAcon ~0.37 vs full xwOBA
~0.31), xwOBAcon (and K/9, whiff%, barrel%, ...) enter as regression FEATURES of
a map g(features) -> runs/9; the coefficient absorbs the absolute scale, so no
cross-scale ratio is ever formed. The label is fit on ACTUAL TOTAL runs allowed
per 9 (not earned) so the output lands directly on the market-graded run scale.

Pure + dependency-free (small ridge OLS via Gaussian elimination) so it unit
-tests without numpy or SQL. The backtest_starters bake-off builds the training
rows from pitcher_asof_daily (as-of features) + actual game runs and calls fit()
on prior seasons, then predict() on the holdout; the live path (candidate-staged,
only if additive wins) applies the same fitted model to the live as-of features.
"""


def _solve(A, b):
    """Solve A x = b for a small square system via Gaussian elimination with partial
    pivoting. Returns the solution list, or None if singular."""
    n = len(A)
    # Augmented copy.
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        # Partial pivot: largest magnitude in this column at/after `col`.
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        pivot = M[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col] / pivot
            if factor:
                for c in range(col, n + 1):
                    M[r][c] -= factor * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit_ols_multi(X, y, ridge=1e-4):
    """Ridge OLS with an intercept. X = list of equal-length feature vectors, y =
    labels. Returns (intercept, [coef_per_feature]) or None. Ridge (small) keeps
    X'X invertible with collinear/thin features and lightly regularizes."""
    if not X or len(X) != len(y):
        return None
    k = len(X[0])
    if any(len(row) != k for row in X):
        return None
    d = k + 1                                    # + intercept column
    # Design with a leading 1 for the intercept.
    Z = [[1.0] + [float(v) for v in row] for row in X]
    # Normal equations (Z'Z + ridge*I) b = Z'y ; do NOT penalize the intercept.
    ztz = [[0.0] * d for _ in range(d)]
    zty = [0.0] * d
    for zi, yi in zip(Z, y):
        for a in range(d):
            zty[a] += zi[a] * yi
            za = zi[a]
            row = ztz[a]
            for bcol in range(d):
                row[bcol] += za * zi[bcol]
    for a in range(1, d):                        # ridge on slopes only
        ztz[a][a] += ridge
    sol = _solve(ztz, zty)
    if sol is None:
        return None
    return sol[0], sol[1:]


def fit(rows, feature_keys, ridge=1e-4):
    """Fit runs/9 = g(features). ``rows`` = dicts carrying ``feature_keys`` + a
    ``label`` (actual total runs allowed per 9). Rows missing any feature or the
    label are dropped. Returns a model dict, or None if too few complete rows.

    model = {feature_keys, intercept, coef, league_rate9 (label mean), n}."""
    xs, ys = [], []
    for r in rows:
        label = r.get("label")
        feats = [r.get(k) for k in feature_keys]
        if label is None or any(f is None for f in feats):
            continue
        xs.append([float(f) for f in feats])
        ys.append(float(label))
    if len(xs) < max(10, 3 * len(feature_keys)):
        return None
    fit_res = fit_ols_multi(xs, ys, ridge=ridge)
    if fit_res is None:
        return None
    intercept, coef = fit_res
    return {"feature_keys": list(feature_keys), "intercept": intercept,
            "coef": coef, "league_rate9": sum(ys) / len(ys), "n": len(ys)}


def predict(feat, model, n_sample=None, prior_strength=150.0,
            lo=1.5, hi=9.0):
    """Predict runs/9 from a feature dict. Missing any model feature -> None (the
    caller falls back to an ERA-based rate). When ``n_sample`` (the pitcher's as-of
    BBE/BF behind the features) is given, the raw prediction is Bayesian-shrunk
    toward the league rate by w = n/(n+prior_strength) — small early-season samples
    lean on the league prior. Clamped to [lo, hi]."""
    if not model:
        return None
    vals = [feat.get(k) for k in model["feature_keys"]]
    if any(v is None for v in vals):
        return None
    raw = model["intercept"] + sum(c * float(v)
                                   for c, v in zip(model["coef"], vals))
    league = model["league_rate9"]
    if n_sample is not None and prior_strength > 0:
        try:
            w = float(n_sample) / (float(n_sample) + prior_strength)
        except (TypeError, ValueError):
            w = 1.0
        raw = w * raw + (1.0 - w) * league
    return max(lo, min(hi, raw))
