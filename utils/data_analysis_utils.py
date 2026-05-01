import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import zscore
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ─── Interpretation helpers ───────────────────────────────────────────────────

def pearson_interpretation(score: float) -> str:
    score_abs = abs(score)
    if score_abs == 0:
        return "no"
    elif score_abs <= 0.3:
        return "weak"
    elif score_abs <= 0.5:
        return "medium"
    elif score_abs <= 1:
        return "strong"
    else:
        return "error"


def f2_interpretation(f2: float) -> str:
    if f2 >= 0.35:
        return "Large"
    elif f2 >= 0.15:
        return "Medium"
    elif f2 >= 0.02:
        return "Small"
    else:
        return "Negligible"


def vif_interpretation(score: float) -> str:
    if score >= 5:
        return "high multicollinearity"
    elif score > 1.5:
        return "acceptable multicollinearity"
    elif score > 1:
        return "minimal multicollinearity"
    elif score == 1:
        return "no multicollinearity"
    else:
        return "something went wrong"


# ─── Multicollinearity ───────────────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame, columns: list) -> None:
    """Print Pearson correlation matrix and interpretation table."""
    df_corr = df[columns].corr(method="pearson")
    try:
        from IPython.display import display
        display(df_corr.round(4))
        display(df_corr.map(pearson_interpretation))
    except ImportError:
        print(df_corr.round(4))
        print(df_corr.map(pearson_interpretation))


def compute_vif(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Return a DataFrame with VIF and interpretation for each predictor."""
    X = sm.add_constant(df[columns].copy())
    vif_values = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    result = pd.DataFrame({
        "variable": X.columns,
        "VIF": vif_values,
        "VIF interpretation": [vif_interpretation(v) for v in vif_values],
    })
    return result[result["variable"].isin(columns)].reset_index(drop=True)


# ─── OLS regression ──────────────────────────────────────────────────────────

def run_ols(
    df: pd.DataFrame,
    predictors: list,
    dependent: str,
    label: str = "",
    network_label: str = "",
):
    """
    Fit OLS and print a structured summary.

    Outputs:
    - R²
    - Descriptive stats for predictors
    - Per-predictor table: Norm. Coef | f² | Interpretation | p-value
    - Full statsmodels summary

    Returns the fitted model.
    """
    X = df[predictors]
    y = df[[dependent]]

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    outcome_label = label or dependent
    header = f"OLS: {outcome_label}"
    if network_label:
        header += f" — {network_label}"

    print("=" * 60)
    print(header)
    print("=" * 60)
    print(f"R² = {model.rsquared:.4f}    Adj. R² = {model.rsquared_adj:.4f}"
          f"    n = {int(model.nobs)}")

    # Descriptive stats
    stats_df = (
        X.agg(["mean", "std"])
        .T
        .rename(columns={"mean": "Mean", "std": "Std"})
    )
    print("-" * 60)
    print("Descriptive statistics (predictors)")
    print("-" * 60)
    try:
        from IPython.display import display
        display(stats_df.round(4))
    except ImportError:
        print(stats_df.round(4))

    # Cohen's f² (drop-one method)
    r2_full = model.rsquared
    f2_values = []
    for col in X.columns:
        X_reduced = sm.add_constant(X.drop(columns=[col]))
        r2_red = sm.OLS(y, X_reduced).fit().rsquared
        f2 = (r2_full - r2_red) / (1 - r2_full + 1e-12)
        f2_values.append(max(f2, 0.0))

    # Standardised coefficients
    X_std = zscore(X, ddof=1)
    y_std = zscore(y, ddof=1)
    model_std = sm.OLS(y_std, sm.add_constant(X_std)).fit()
    coef_std = model_std.params[1:]

    predictors_df = pd.DataFrame({
        "Norm. Coef": coef_std.values,
        "f²": f2_values,
        "f² Interpretation": [f2_interpretation(v) for v in f2_values],
        "p-value": model.pvalues[1:].round(4).values,
    }, index=X.columns)
    predictors_df = predictors_df.sort_values("f²", ascending=False)

    print("-" * 60)
    print("Predictor strength")
    print("-" * 60)
    try:
        from IPython.display import display
        display(predictors_df.round(4))
    except ImportError:
        print(predictors_df.round(4))

    print("-" * 60)
    print("Full summary")
    print("-" * 60)
    print(model.summary())

    return model


# ─── Diagnostics ─────────────────────────────────────────────────────────────

def regression_diagnostics(model, title: str = "") -> None:
    """2×2 diagnostic grid: Residuals vs Fitted, Q-Q, Scale-Location, Leverage."""
    fitted = model.fittedvalues
    resid = model.resid
    influence = model.get_influence()
    std_resid = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag

    fig, axs = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=12)

    # Residuals vs Fitted
    axs[0, 0].scatter(fitted, resid, color="#2980b9", alpha=0.4, s=8)
    axs[0, 0].axhline(0, color="black", linewidth=1)
    axs[0, 0].set_xlabel("Fitted values")
    axs[0, 0].set_ylabel("Residuals")
    axs[0, 0].set_title("Residuals vs Fitted")

    # Normal Q-Q
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm", plot=None)
    axs[0, 1].scatter(osm, osr, color="#2980b9", alpha=0.4, s=8)
    axs[0, 1].plot(osm, slope * osm + intercept, color="black", linewidth=1)
    axs[0, 1].set_title("Normal Q-Q")
    axs[0, 1].set_xlabel("Theoretical quantiles")
    axs[0, 1].set_ylabel("Ordered values")

    # Scale-Location
    axs[1, 0].scatter(fitted, np.sqrt(np.abs(std_resid)), color="#2980b9", alpha=0.4, s=8)
    axs[1, 0].set_xlabel("Fitted values")
    axs[1, 0].set_ylabel("√|Standardised residuals|")
    axs[1, 0].set_title("Scale-Location")

    # Residuals vs Leverage
    axs[1, 1].scatter(leverage, std_resid, color="#2980b9", alpha=0.4, s=8)
    axs[1, 1].axhline(0, color="black", linewidth=1)
    axs[1, 1].set_xlabel("Leverage")
    axs[1, 1].set_ylabel("Standardised residuals")
    axs[1, 1].set_title("Residuals vs Leverage")

    plt.show()


# ─── Effect-size comparison chart ────────────────────────────────────────────

def plot_f2_comparison(
    models: dict,
    predictors: list,
    title: str = "Cohen's f² — Effect Size Comparison",
) -> None:
    """
    Side-by-side grouped bar chart of Cohen's f² per predictor across outcomes.

    Parameters
    ----------
    models : dict mapping outcome label → fitted OLS model
    predictors : list of predictor names (must match model column order)
    """
    n_outcomes = len(models)
    n_pred = len(predictors)
    x = np.arange(n_pred)
    width = 0.8 / n_outcomes
    colors = plt.cm.tab10(np.linspace(0, 0.5, n_outcomes))

    fig, ax = plt.subplots(figsize=(max(8, n_pred * 2), 5))

    for i, (outcome_label, model) in enumerate(models.items()):
        r2_full = model.rsquared
        X_cols = [c for c in model.model.exog_names if c != "const"]
        X_df = model.model.exog[:, 1:]  # drop constant column
        X_frame = pd.DataFrame(X_df, columns=X_cols)
        y_frame = pd.DataFrame(model.model.endog)

        f2_vals = []
        for col in predictors:
            if col not in X_cols:
                f2_vals.append(0.0)
                continue
            X_red = sm.add_constant(X_frame.drop(columns=[col]))
            r2_red = sm.OLS(y_frame, X_red).fit().rsquared
            f2 = (r2_full - r2_red) / (1 - r2_full + 1e-12)
            f2_vals.append(max(f2, 0.0))

        offset = (i - n_outcomes / 2 + 0.5) * width
        bars = ax.bar(x + offset, f2_vals, width, label=outcome_label,
                      color=colors[i], alpha=0.85)
        for bar, val in zip(bars, f2_vals):
            if val > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # Threshold lines
    for threshold, linelabel in [(0.02, "Small"), (0.15, "Medium"), (0.35, "Large")]:
        ax.axhline(threshold, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(n_pred - 0.1, threshold + 0.005, linelabel,
                ha="right", va="bottom", fontsize=8, color="grey")

    ax.set_xticks(x)
    ax.set_xticklabels(predictors, fontsize=10)
    ax.set_ylabel("Cohen's f²", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()
