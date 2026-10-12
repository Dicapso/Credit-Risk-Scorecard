"""Synthetic consumer-loan data for a Baku microfinance lender (fixed seed).

Writes two files next to this script:
  loans.csv               booked loans Jan 2024 - Sep 2025 with a 12-month outcome
                          (bad = 60+ days past due within 12 months on book)
  applications_recent.csv through-the-door applications Apr - Sep 2026 (no outcome yet),
                          with a deliberate shift in channel mix and applicant profile

A few realistic data issues are injected so the cleaning step has work to do.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
OUT = Path(__file__).resolve().parent
APR = 0.24


def annuity(amount, term, apr=APR):
    r = apr / 12
    return amount * r / (1 - (1 + r) ** -term)


def simulate(n, rng, start, end, recent=False):
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    app_date = pd.Timestamp(start) + pd.to_timedelta(np.sort(rng.integers(0, days + 1, n)), unit="D")

    # channel mix shifts towards online in the recent period
    ch_p = [0.33, 0.45, 0.22] if recent else [0.45, 0.30, 0.25]
    channel = rng.choice(["branch", "online", "partner_store"], n, p=ch_p)

    region = rng.choice(["Baku", "Sumqayit/Absheron", "Ganja", "Other regions"], n, p=[0.55, 0.15, 0.10, 0.20])
    age_mu = np.where(channel == "online", 31, 39) - (2 if recent else 0)
    age = np.clip(rng.normal(age_mu, 10), 21, 68).round().astype(int)

    emp = rng.choice(["salaried_public", "salaried_private", "self_employed", "informal"], n,
                     p=[0.24, 0.42, 0.17, 0.17])
    emp = np.where(age >= 63, "pensioner", np.where((age >= 58) & (rng.random(n) < 0.5), "pensioner", emp))
    emp_years = np.minimum(rng.gamma(2.0, np.clip((age - 20) / 6, 0.5, None)), age - 18).round(1)
    emp_years = np.where(emp == "pensioner", 0, emp_years)

    base_inc = pd.Series(emp).map({"salaried_public": 850, "salaried_private": 1050, "self_employed": 1150,
                                   "informal": 650, "pensioner": 420}).to_numpy()
    reg_mult = pd.Series(region).map({"Baku": 1.15, "Sumqayit/Absheron": 0.95, "Ganja": 0.88,
                                      "Other regions": 0.80}).to_numpy()
    income = np.round(base_inc * reg_mult * rng.lognormal(0, 0.38, n) * (1 + 0.006 * np.clip(age - 21, 0, 25)), -1)
    salary_card = ((np.isin(emp, ["salaried_public", "salaried_private"])) & (rng.random(n) < 0.45)).astype(int)

    purpose = np.where(channel == "partner_store", "consumer_goods",
                       rng.choice(["cash", "home_repair", "car", "education", "consumer_goods"], n,
                                  p=[0.45, 0.20, 0.12, 0.08, 0.15]))
    amt_mult = pd.Series(purpose).map({"cash": 2.2, "home_repair": 3.5, "car": 7.0, "education": 2.8,
                                       "consumer_goods": 1.4}).to_numpy()
    loan_amount = np.clip(np.round(income * amt_mult * rng.lognormal(0, 0.45, n), -1), 300, 30000)
    term = np.where(loan_amount > 6000, rng.choice([24, 36], n, p=[0.5, 0.5]),
                    rng.choice([6, 12, 18, 24], n, p=[0.2, 0.4, 0.25, 0.15]))

    # hidden creditworthiness; the bureau score sees part of it
    z = rng.normal(0, 1, n)
    thin_p = np.clip(0.06 + 0.25 * (age < 26) + 0.12 * (emp == "informal") + 0.05 * (channel == "online")
                     + (0.04 if recent else 0), 0, 0.9)
    thin = rng.random(n) < thin_p
    bureau = np.clip(np.round(610 + 70 * z + rng.normal(0, 45, n)), 300, 850)
    bureau = np.where(thin, np.nan, bureau)
    existing_loans = np.where(thin, 0, rng.poisson(0.9 + 0.3 * (z < -0.5), n))
    delinq_12m = np.where(thin, 0, rng.poisson(np.exp(-1.6 - 0.8 * z), n))
    existing_pmt = existing_loans * rng.uniform(60, 220, n)
    # old affordability rule: total instalments may not exceed 80% of income -> shrink the loan
    max_pmt = np.clip(0.80 * income - existing_pmt, annuity(300, term), None)
    loan_amount = np.maximum(np.minimum(loan_amount, np.floor(loan_amount * max_pmt / annuity(loan_amount, term) / 10) * 10), 300)
    dti = np.round((existing_pmt + annuity(loan_amount, term)) / income, 3)

    out = pd.DataFrame({
        "application_date": app_date.strftime("%Y-%m-%d"), "channel": channel, "region": region,
        "age": age, "employment_type": emp, "employment_years": emp_years,
        "monthly_income_azn": income, "salary_card_client": salary_card, "loan_purpose": purpose,
        "loan_amount_azn": loan_amount, "term_months": term, "existing_loans": existing_loans,
        "delinquencies_12m": delinq_12m, "bureau_score": bureau, "debt_to_income": dti,
    })

    emp_eff = pd.Series(emp).map({"salaried_public": -0.45, "salaried_private": 0.0, "self_employed": 0.35,
                                  "informal": 0.65, "pensioner": -0.55}).to_numpy()
    logit = (-3.15 - 0.85 * z + emp_eff
             + 0.55 * np.clip(30 - age, 0, None) / 9 + 0.12 * np.clip(age - 55, 0, None) / 5
             + 2.2 * np.clip(dti - 0.35, 0, None) + 0.35 * np.clip(dti - 0.6, 0, None) * 3
             - 0.55 * salary_card + 0.40 * (channel == "online") + 0.15 * (channel == "partner_store")
             + 0.30 * np.minimum(delinq_12m, 4) + 0.45 * thin + 0.30 * (term >= 36)
             + 0.25 * (purpose == "cash") - 0.20 * (purpose == "car")
             + 0.18 * (region == "Other regions") - 0.25 * np.log(income / 900)
             - 0.03 * np.minimum(emp_years, 10))
    # slightly tougher market for loans booked from Q2 2025
    logit = logit + 0.18 * (app_date >= pd.Timestamp("2025-04-01"))
    return out, 1 / (1 + np.exp(-logit))


def main():
    rng = np.random.default_rng(SEED)
    loans, pd_true = simulate(24_000, rng, "2024-01-02", "2025-09-30")
    bad = (rng.random(len(loans)) < pd_true).astype(int)
    loans["bad_60dpd_12m"] = bad
    mob = rng.choice(np.arange(2, 13), len(loans), p=np.array([5, 9, 12, 13, 12, 11, 10, 9, 7, 6, 6]) / 100)
    loans["months_to_default"] = np.where(bad == 1, mob, np.nan)
    loans.insert(0, "loan_id", [f"LN{100000 + i}" for i in range(len(loans))])

    # --- inject data-quality issues -------------------------------------------------
    idx = rng.choice(len(loans), 260, replace=False)
    loans.loc[idx[:150], "monthly_income_azn"] = np.nan                      # income not captured
    loans.loc[idx[150:170], "monthly_income_azn"] *= 100                      # keyed in qəpik
    loans.loc[idx[170:176], "age"] = rng.choice([0, 7, 199, 230], 6)          # impossible ages
    loans.loc[idx[176:200], "employment_years"] = loans.loc[idx[176:200], "age"] + 5  # longer than adult life
    dup = loans.sample(45, random_state=SEED)
    loans = pd.concat([loans, dup]).sort_values(["application_date", "loan_id"]).reset_index(drop=True)
    loans.to_csv(OUT / "loans.csv", index=False)

    recent, _ = simulate(6_000, rng, "2026-04-01", "2026-09-30", recent=True)
    recent.insert(0, "application_id", [f"AP{500000 + i}" for i in range(len(recent))])
    recent.to_csv(OUT / "applications_recent.csv", index=False)
    print(f"loans: {len(loans):,} rows, bad rate {loans.bad_60dpd_12m.mean():.2%}; recent: {len(recent):,}")


if __name__ == "__main__":
    main()
