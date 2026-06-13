"""Hypothesis tests for the STAT 250 OECD macro indicators project."""

import pandas as pd
import pingouin as pg # for Welch ANOVA and Games-Howell post-hoc
from scipy import stats
from scipy.stats import f_oneway, ttest_1samp, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest

from .features import PERIOD_ORDER

def _first_available_value(row: pd.Series, columns: tuple[str, ...]) -> float:
    """
    Return the first matching value from version-dependent output columns. ('p-unc' vs 'p_unc' in pingouin)
    """
    for column in columns:
        if column in row.index:
            return row[column]
    raise KeyError(f"None of the expected columns were found: {columns}")

def run_rq1_turkiye_unemployment(panel: pd.DataFrame, reference: float = 6.5) -> dict:
    """
    RQ1: one-sample t-test for Turkiye's average unemployment.
    """
    sample = panel.loc[panel["country"] == "TUR", "unemp"].dropna()
    t_stat, p_value = ttest_1samp(sample, popmean=reference)
    ci = stats.t.interval(0.95, len(sample) - 1, loc=sample.mean(), scale=sample.sem())
    return {
        "rq": "RQ1",
        "test": "One-sample t-test",
        "n": len(sample),
        "mean": sample.mean(),
        "reference": reference,
        "statistic": t_stat,
        "p_value": p_value,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "decision": "Reject H0" if p_value < 0.05 else "Fail to reject H0",
    }

def run_rq2_covid_ipi(panel: pd.DataFrame) -> dict:
    """
    RQ2: Turkiye vs other OECD IPI during COVID-19.
    """
    covid = panel[panel["period"].astype(str) == "COVID-19 2020"]
    tur = covid.loc[covid["country"] == "TUR", "ipi"].dropna()
    other = covid.loc[covid["country"] != "TUR", "ipi"].dropna()
    levene_stat, levene_p = stats.levene(tur, other)
    equal_var = bool(levene_p >= 0.05)
    t_stat, p_value = ttest_ind(tur, other, equal_var=equal_var)
    return {
        "rq": "RQ2",
        "test": "Student t-test" if equal_var else "Welch t-test",
        "levene_statistic": levene_stat,
        "levene_p_value": levene_p,
        "n_turkiye": len(tur),
        "n_other_oecd": len(other),
        "mean_turkiye": tur.mean(),
        "mean_other_oecd": other.mean(),
        "statistic": t_stat,
        "p_value": p_value,
        "decision": "Reject H0" if p_value < 0.05 else "Fail to reject H0",
    }

def run_rq3_turkiye_high_inflation(panel: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    RQ3: one-sample proportion z-test for Turkiye's high-inflation quarters.
    """
    tur = panel[(panel["country"] == "TUR") & panel["cpi_qoq"].notna()].copy()
    successes = int(tur["high_inflation"].sum())
    nobs = int(tur.shape[0])
    z_stat, p_value = proportions_ztest(successes, nobs, value=0.50, alternative="larger")

    bins = [(2005, 2009), (2010, 2014), (2015, 2019), (2020, 2025)]
    rows = []
    for start, end in bins:
        subset = tur[tur["year"].between(start, end)]
        rows.append(
            {
                "subperiod": f"{start}-{end}",
                "n": len(subset),
                "high_inflation_quarters": int(subset["high_inflation"].sum()),
                "proportion": subset["high_inflation"].mean(),
            }
        )
    breakdown = pd.DataFrame(rows)

    result = {
        "rq": "RQ3",
        "test": "One-sample proportion z-test",
        "n": nobs,
        "high_inflation_quarters": successes,
        "sample_proportion": successes / nobs,
        "reference": 0.50,
        "statistic": z_stat,
        "p_value": p_value,
        "decision": "Reject H0" if p_value < 0.05 else "Fail to reject H0",
    }
    return result, breakdown

def run_rq4_turkiye_vs_oecd_high_inflation(panel: pd.DataFrame) -> dict:
    """
    RQ4: two-sample proportion z-test for Turkiye vs other OECD high-inflation ratios.
    """
    inf = panel[panel["cpi_qoq"].notna()].copy()
    tur = inf[inf["country"] == "TUR"]
    other = inf[inf["country"] != "TUR"]
    successes = [int(tur["high_inflation"].sum()), int(other["high_inflation"].sum())]
    nobs = [int(len(tur)), int(len(other))]
    z_stat, p_value = proportions_ztest(successes, nobs, alternative="two-sided")
    return {
        "rq": "RQ4",
        "test": "Two-sample proportion z-test",
        "n_turkiye": nobs[0],
        "n_other_oecd": nobs[1],
        "proportion_turkiye": successes[0] / nobs[0],
        "proportion_other_oecd": successes[1] / nobs[1],
        "statistic": z_stat,
        "p_value": p_value,
        "decision": "Reject H0" if p_value < 0.05 else "Fail to reject H0",
    }

def run_rq5_anova_unemployment_by_period(panel: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    RQ5: unemployment differences across macroeconomic periods.
    """
    data = panel[["period", "unemp", "country"]].dropna().copy()
    data["period"] = pd.Categorical(data["period"], categories=PERIOD_ORDER, ordered=True)
    data = data.dropna(subset=["period"])
    group_stats = (
        data.groupby("period", observed=True)["unemp"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    groups = [data.loc[data["period"] == period, "unemp"].to_numpy() for period in PERIOD_ORDER]
    groups = [group for group in groups if len(group) > 0]

    levene_stat, levene_p = stats.levene(*groups)
    classic_f, classic_p = f_oneway(*groups)

    welch_table = pg.welch_anova(
        data=data,
        dv="unemp",
        between="period"
    )

    welch_row = welch_table.loc[0]
    welch_f = welch_row["F"]
    welch_df1 = welch_row["ddof1"]
    welch_df2 = welch_row["ddof2"]
    welch_p = _first_available_value(welch_row, ("p-unc", "p_unc"))
    
    main_test = "Welch ANOVA" if levene_p < 0.05 else "Classic one-way ANOVA"
    main_p = welch_p if levene_p < 0.05 else classic_p

    result = {
        "rq": "RQ5",
        "test": main_test,
        "levene_statistic": levene_stat,
        "levene_p_value": levene_p,
        "classic_f": classic_f,
        "classic_p_value": classic_p,
        "welch_f": welch_f,
        "welch_df1": welch_df1,
        "welch_df2": welch_df2,
        "welch_p_value": welch_p,
        "statistic": welch_f if levene_p < 0.05 else classic_f,
        "p_value": main_p,
        "decision": "Reject H0" if main_p < 0.05 else "Fail to reject H0",
    }

    if levene_p >= 0.05:
        tukey = pairwise_tukeyhsd(data["unemp"], data["period"], alpha=0.05)
        posthoc = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    else:
        posthoc = pg.pairwise_gameshowell(data=data, dv="unemp", between="period")
    return result, group_stats, posthoc

def run_all_hypothesis_tests(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Run RQ1-RQ5 and return summary plus supporting tables.
    """
    rq1 = run_rq1_turkiye_unemployment(panel)
    rq2 = run_rq2_covid_ipi(panel)
    rq3, rq3_breakdown = run_rq3_turkiye_high_inflation(panel)
    rq4 = run_rq4_turkiye_vs_oecd_high_inflation(panel)
    rq5, rq5_group_stats, rq5_posthoc = run_rq5_anova_unemployment_by_period(panel)

    summary = pd.DataFrame([rq1, rq2, rq3, rq4, rq5])
    supporting = {
        "rq3_high_inflation_subperiods": rq3_breakdown,
        "rq5_anova_group_stats": rq5_group_stats,
        "rq5_posthoc": rq5_posthoc,
    }
    return summary, supporting
