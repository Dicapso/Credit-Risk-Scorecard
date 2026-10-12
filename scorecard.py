"""Cleaning, WoE binning, scorecard scaling and validation metrics for the Day 11 project."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
TARGET = "bad_60dpd_12m"
NUMERIC = ["age", "employment_years", "monthly_income_azn", "loan_amount_azn", "term_months",
           "existing_loans", "delinquencies_12m", "bureau_score", "debt_to_income"]
CATEGORICAL = ["channel", "region", "employment_type", "loan_purpose", "salary_card_client"]


def load_raw():
    loans = pd.read_csv(DATA / "loans.csv", parse_dates=["application_date"])
    recent = pd.read_csv(DATA / "applications_recent.csv", parse_dates=["application_date"])
    return loans, recent


def clean(df):
    """Fix the known data issues. Returns the cleaned frame and a log of what changed."""
    df = df.copy()
    log = {"Rows in": len(df)}
    n = len(df)
    df = df.drop_duplicates()
    log["Exact duplicate rows dropped"] = n - len(df)
    qepik = df.monthly_income_azn > 20_000
    df.loc[qepik, "monthly_income_azn"] /= 100
    log["Incomes keyed in qəpik (÷100)"] = int(qepik.sum())
    bad_age = ~df.age.between(18, 80)
    df.loc[bad_age, "age"] = np.nan
    log["Impossible ages set to missing"] = int(bad_age.sum())
    bad_emp = df.employment_years > (df.age.fillna(99) - 16)
    df.loc[bad_emp, "employment_years"] = np.nan
    log["Employment years > adult life set to missing"] = int(bad_emp.sum())
    log["Missing income (kept as own WoE bin)"] = int(df.monthly_income_azn.isna().sum())
    log["Missing bureau score = thin file (own bin)"] = int(df.bureau_score.isna().sum())
    log["Rows out"] = len(df)
    return df.reset_index(drop=True), log


# ---------------------------------------------------------------- WoE binning
def _woe_table(x_bin, y):
    t = pd.DataFrame({"bin": x_bin, "y": y}).groupby("bin", observed=True, sort=False).y.agg(["size", "sum"])
    t.columns = ["n", "bad"]
    t["good"] = t.n - t.bad
    g, b = t.good.sum(), t.bad.sum()
    t["dist_good"] = (t.good + 0.5) / g
    t["dist_bad"] = (t.bad + 0.5) / b
    t["bad_rate"] = t.bad / t.n
    t["woe"] = np.log(t.dist_good / t.dist_bad)
    if "Missing" in t.index and t.loc["Missing", "n"] < 0.01 * t.n.sum():
        t.loc["Missing", "woe"] = 0.0                    # too few rows to trust: treat as neutral
    t["iv"] = (t.dist_good - t.dist_bad) * t.woe
    return t


def bin_numeric(x, y, max_bins=6, n_init=12, min_share=0.05):
    """Monotonic binning: start from quantile bins, merge adjacent bins until the bad rate is
    monotonic, every bin holds >= min_share of rows and there are at most max_bins bins.
    Missing values get their own bin. Returns the inner cut points."""
    m = x.notna()
    uniq = np.sort(x[m].unique())
    if len(uniq) <= n_init:                                 # discrete variable: one bin per value
        edges = uniq[1:]
    else:
        edges = np.unique(np.quantile(x[m], np.linspace(0, 1, n_init + 1))[1:-1])
    while True:
        cuts = np.r_[-np.inf, edges, np.inf]
        b = pd.cut(x[m], cuts, labels=False, right=False)
        st = pd.DataFrame({"b": b, "y": y[m]}).groupby("b").y.agg(["size", "mean"]).reindex(range(len(cuts) - 1))
        if len(st) <= 1:
            return list(edges)
        rate, size = st["mean"].to_numpy(), st["size"].to_numpy()
        direction = np.sign(np.corrcoef(np.arange(len(rate)), rate)[0, 1]) or 1
        diffs = np.diff(rate) * direction
        small = size < min_share * m.sum()
        if (diffs < 0).any():
            i = int(np.argmin(diffs))                       # worst violation: merge bins i and i+1
        elif small.any():
            j = int(np.argmax(small))
            i = j - 1 if j == len(rate) - 1 else (j if j == 0 else (j - 1 if abs(rate[j] - rate[j - 1]) < abs(rate[j + 1] - rate[j]) else j))
        elif len(rate) > max_bins:
            i = int(np.argmin(np.abs(np.diff(rate))))
        else:
            return list(edges)
        edges = np.delete(edges, i)


class WoEBinner:
    def __init__(self, max_bins=6):
        self.max_bins = max_bins
        self.spec, self.tables = {}, {}

    def _assign(self, col, x):
        kind, val = self.spec[col]
        if kind == "num":
            cuts = np.r_[-np.inf, val, np.inf]
            labels = [f"[{_fmt(cuts[i])}, {_fmt(cuts[i + 1])})" for i in range(len(cuts) - 1)]
            out = pd.cut(x, cuts, labels=labels, right=False).astype(object)
        else:
            out = x.astype(str).map(lambda v: val.get(v, "OTHER"))
        return out.where(x.notna(), "Missing")

    def fit(self, X, y):
        for col in X.columns:
            x = X[col]
            if col in NUMERIC:
                self.spec[col] = ("num", bin_numeric(x, y, self.max_bins))
            else:
                # categories under 3% of rows are grouped into OTHER
                share = x.astype(str).value_counts(normalize=True)
                self.spec[col] = ("cat", {k: (k if s >= 0.03 else "OTHER") for k, s in share.items()})
            t = _woe_table(self._assign(col, x), y)
            if self.spec[col][0] == "num":
                order = [l for l in self._labels(col) if l in t.index]
                t = t.reindex(order + (["Missing"] if "Missing" in t.index else []))
            else:
                t = t.sort_values("woe")
            self.tables[col] = t
        return self

    def _labels(self, col):
        cuts = np.r_[-np.inf, self.spec[col][1], np.inf]
        return [f"[{_fmt(cuts[i])}, {_fmt(cuts[i + 1])})" for i in range(len(cuts) - 1)]

    def transform(self, X):
        out = {}
        for col in X.columns:
            t = self.tables[col]
            out[col] = self._assign(col, X[col]).map(t.woe).fillna(0.0).astype(float)
        return pd.DataFrame(out, index=X.index)

    def bins(self, X):
        return pd.DataFrame({c: self._assign(c, X[c]) for c in X.columns}, index=X.index)

    def iv(self):
        return pd.Series({c: t.iv.sum() for c, t in self.tables.items()}).sort_values(ascending=False)


def _fmt(v):
    if np.isinf(v):
        return "-inf" if v < 0 else "inf"
    return f"{v:,.3g}" if abs(v) < 10 else f"{v:,.0f}"


# ---------------------------------------------------------------- scaling & metrics
def scaling(pdo=20, base_score=600, base_odds=50):
    factor = pdo / np.log(2)
    return factor, base_score - factor * np.log(base_odds)


def points_table(binner, features, coef, intercept, pdo=20, base_score=600, base_odds=50):
    """Points per attribute so that score = sum of points = offset + factor * ln(odds good)."""
    factor, offset = scaling(pdo, base_score, base_odds)
    n = len(features)
    rows = []
    for f, b in zip(features, coef):
        t = binner.tables[f]
        for lab, r in t.iterrows():
            pts = -(b * r.woe + intercept / n) * factor + offset / n
            rows.append({"feature": f, "bin": lab, "n": int(r.n), "bad_rate": r.bad_rate, "woe": r.woe, "points": pts})
    return pd.DataFrame(rows)


def score_from_logit(logit_bad, pdo=20, base_score=600, base_odds=50):
    factor, offset = scaling(pdo, base_score, base_odds)
    return offset - factor * logit_bad


def ks_stat(y, score):
    """KS between the score distributions of goods and bads (higher score = safer)."""
    d = pd.DataFrame({"y": y, "s": score}).sort_values("s")
    cb = d.y.cumsum() / d.y.sum()
    cg = (1 - d.y).cumsum() / (1 - d.y).sum()
    gap = (cb - cg).to_numpy()
    i = int(np.argmax(gap))
    return gap[i], d.s.iloc[i]


def psi(expected, actual, edges):
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e))), e, a


def psi_categorical(expected, actual):
    e = expected.value_counts(normalize=True)
    a = actual.value_counts(normalize=True).reindex(e.index.union(actual.unique())).fillna(0)
    e = e.reindex(a.index).fillna(0)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def loan_profit(amount, term, bad, net_margin=0.13, ead=0.80, lgd=0.70):
    """Simple unit economics per booked loan, AZN.
    good: net interest margin (after funding & opex) on the average balance (~55% of amount) over the term
    bad : exposure at default x loss given default"""
    good_profit = amount * net_margin * 0.55 * term / 12
    return np.where(bad == 1, -amount * ead * lgd, good_profit)
