"""Ordinary least squares (OLS) regression with HAC/Newey-West standard errors and diagnostics on Turkiye data."""

import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson


FULL_MODEL_VARIABLES = ["cpi_yoy", "ipi_yoy", "cci", "gdp_growth", "trend"]


def _fit_ols_hac(data: pd.DataFrame, y_col: str, x_cols: list[str], maxlags: int = 4):
    """
    Fit an OLS regression with HAC/Newey-West standard errors, falling back to HC3 if HAC fails.
    """
    X = sm.add_constant(data[x_cols])
    y = data[y_col]
    try:
        return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags}), "HAC/Newey-West"
    except Exception:
        return sm.OLS(y, X).fit(cov_type="HC3"), "HC3" # fallback to HC3 standard errors if HAC fails (e.g., due to small sample size)


def _diagnostics(model, X) -> dict:
    """
    Calculate regression diagnostics: Shapiro-Wilk test for normality of residuals, 
    Breusch-Pagan test for heteroskedasticity, and 
    Durbin-Watson statistic for autocorrelation of residuals.
    """
    resid = model.resid
    shapiro_n = min(50, len(resid))
    shapiro_stat, shapiro_p = stats.shapiro(resid[:shapiro_n]) # Shapiro-Wilk test for normality of residuals, using up to the first 50 residuals due to test limitations.
    bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(resid, X) #heteroskedasticity test: null hypothesis is homoskedasticity, alternative is heteroskedasticity.
    return {
        "residual_shapiro_statistic": shapiro_stat,
        "residual_shapiro_p_value": shapiro_p,
        "breusch_pagan_lm": bp_lm,
        "breusch_pagan_p_value": bp_p,
        "breusch_pagan_f": bp_f,
        "breusch_pagan_f_p_value": bp_fp,
        "durbin_watson": durbin_watson(resid), # Durbin-Watson statistic for autocorrelation of residuals, with values around 2 suggesting no autocorrelation, values < 2 suggesting positive autocorrelation, and values > 2 suggesting negative autocorrelation.
    }


def fit_turkiye_regressions(regression_data: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Fit the simple and full Turkiye unemployment regressions.
    """
    turkiye = regression_data[regression_data["country"] == "TUR"].copy()
    if turkiye.empty:
        raise ValueError("No Turkiye observations are available in regression_data.")

    simple_data = turkiye.dropna(subset=["unemp", "ipi_yoy"]).copy()
    full_data = turkiye.dropna(subset=["unemp", *FULL_MODEL_VARIABLES]).copy()

    simple_model, simple_cov = _fit_ols_hac(simple_data, "unemp", ["ipi_yoy"])
    full_model, full_cov = _fit_ols_hac(full_data, "unemp", FULL_MODEL_VARIABLES)

    simple_X = sm.add_constant(simple_data[["ipi_yoy"]])
    full_X = sm.add_constant(full_data[FULL_MODEL_VARIABLES])

    models = {
        "simple": simple_model,
        "full": full_model,
        "simple_data": simple_data,
        "full_data": full_data,
    }
    summary = pd.DataFrame(
        [
            {
                "model": "Simple: unemp ~ ipi_yoy",
                "n": int(simple_model.nobs),
                "r_squared": simple_model.rsquared,
                "adj_r_squared": simple_model.rsquared_adj,
                "f_p_value": simple_model.f_pvalue,
                "covariance": simple_cov,
                **_diagnostics(simple_model, simple_X),
            },
            {
                "model": "Full: unemp ~ cpi_yoy + ipi_yoy + cci + gdp_growth + trend",
                "n": int(full_model.nobs),
                "r_squared": full_model.rsquared,
                "adj_r_squared": full_model.rsquared_adj,
                "f_p_value": full_model.f_pvalue,
                "covariance": full_cov,
                **_diagnostics(full_model, full_X),
            },
        ]
    )

    coef_rows = []
    for label, model in [("simple", simple_model), ("full", full_model)]:
        conf = model.conf_int()
        for term in model.params.index:
            coef_rows.append(
                {
                    "model": label,
                    "term": term,
                    "coef": model.params[term],
                    "std_err": model.bse[term],
                    "p_value": model.pvalues[term],
                    "ci_low": conf.loc[term, 0],
                    "ci_high": conf.loc[term, 1],
                }
            )
    coefficients = pd.DataFrame(coef_rows)
    return models, summary, coefficients

