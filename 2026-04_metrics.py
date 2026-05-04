# %%
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist

import matplotlib.path as mpath
import numpy as np

plt.rcParams.update({"font.size": 10})
bigfs = 9
smallfs = 7
plt.rcParams.update({"font.size": bigfs})
plt.rcParams.update({"axes.titlesize": bigfs})
plt.rcParams.update({"axes.labelsize": bigfs})
plt.rcParams.update({"legend.fontsize": smallfs})
plt.rcParams.update({"xtick.labelsize": smallfs})
plt.rcParams.update({"ytick.labelsize": smallfs})


cmapS = dict(
    zip(
        [1, 2, 4, 8, 16],
        [
            (0.0000, 0.4470, 0.7410),
            (0.8500, 0.3250, 0.0980),
            (0.9290, 0.6940, 0.1250),
            (0.4940, 0.1840, 0.5560),
            (0.4660, 0.6740, 0.1880),
            (0.3010, 0.7450, 0.9330),
            (0.6350, 0.0780, 0.1840),
        ],
    )
)

metric2title = dict(
    x_focal=r"$x_{foc}$",
    extr_nonfoc=r"$|X_{non\text{-}foc}|$",
    n_nbs=r"$|\mathcal{K}|$",
    absOm_tot=r"BN-$|\Omega|$",
    absOm_foc=r"BN-$|\Omega_{foc}|$",
    tb_tot=r"BN-$\alpha$",
    tb_foc=r"BN-$\alpha_{foc}$",
    clust=r"BN-clust",
    bc=r"BN-centr$_{foc}$",
    # bn_expected_influence = r"$\langle\delta x_{foc}\rangle$",
    Hpersfoc=r"$D_\mathrm{BN\text{-}foc}$",
    Hpersnonfoc=r"$D_\mathrm{BN\text{-}non\text{-}foc}$",
    Hsoc=r"$D_{social}$",
    # external_energy = r"$D_{ext}$",
    # energy = r"$D_{tot}$",
)
metric2titleVerb = dict(
    x_focal=r"focal belief",
    extr_nonfoc=r"extremity non-focal beliefs",
    n_nbs=r"nr social contacts",
    absOm_tot=r"connectedness BN",
    absOm_foc=r"connectedness focal BN",
    tb_tot=r"balance BN",
    tb_foc=r"balance focal BN",
    clust=r"clustering BN",
    bc=r"BN focal centrality",
    Hpersfoc=r"focal BN dissonance",
    Hpersnonfoc=r"non-focal BN dissonance",
    Hsoc=r"focal social dissonance",
)

response_cols = [
    "persistPos",
    "nonpersistPos",
    "compliant",
    "resilient",
    "resistant",
    "latecompliant",
]
negcols = ["compliant", "resilient", "resistant"]


belief_dimensions = [int(a) for a in range(10)]
belief_columns = [str(int(a)) for a in belief_dimensions]
edgelist = list(combinations(belief_dimensions, 2))
edges_columns = [f"w{a}{b}" for a, b in edgelist]
metric_cols = [
    "n_nbs",
    "Hpers",
    "Hpersfoc",
    "Hsoc",
    "Hext",
    "tb_tot",
    "tb_foc",
    "absOm_tot",
    "absOm_foc",
    "clust",
    "bc",
    "expI",
    "x_focal",
    "extr_nonfoc",
]
metric_cols2 = [
    "n_nbs",
    "Hpersnonfoc",
    "Hpersfoc",
    "Hsoc",
    "tb_tot",
    "tb_foc",
    "absOm_tot",
    "absOm_foc",
    "clust",
    "bc",
    "x_focal",
    "extr_nonfoc",
]

meta_cols = ["t", "id"]

cmap = dict(
    zip(
        response_cols + ["NA"],
        ["#4CAF50", "#AED581", "#2196F3", "#9C27B0", "#F44336", "#90CAF9", "#9E9E9E"],
    )
)
names = {
    (0.2, 0.0, 0.0, False): r"static ($\omega_0=0.2$)",
    (0.2, 1.0, 0.0, False): r"adaptive",
    (0.8, 0.0, 0.0, False): r"static ($\omega_0=0.8$)",
    (0.2, 1.0, 0.0, True): r"adaptive$\rightarrow$static",
}


# %%

# ----------------------------------------------
# -------    LOAD DATA
# ----------------------------------------------
res = []
mean_absedges = []
init_w, eps, mu, fixedBNat100 = (0.2, 1.0, 0.0, False)
res = pd.DataFrame()
all_pressures = [1, 2, 4, 8, 16]
for s in all_pressures:
    for seed in range(100):
        df = pd.read_csv(
            f"simOut/sim_link_prob0.10_init_w{init_w:.2f}_beta3.00_rho0.33_eps{eps:.2f}_mu{mu:.2f}{'_fixedBNat100' if fixedBNat100 else ''}_ext_strength{s}_seed{seed}.csv"
        )
        W = df.loc[df.t == 100, edges_columns].values
        dists = pdist(W, metric="cityblock")
        groupishness = 0 if eps == 0 else dists.std() / dists.mean()
        responses = (
            df.loc[df.t == 95.5][["id"] + response_cols]
            .melt(id_vars=["id"])
            .replace({False: np.nan})
            .dropna()
        )

        vals = df.loc[df.t == 95.5][meta_cols + metric_cols].reset_index()
        vals["response"] = None
        for col in response_cols:
            vals.loc[
                (df.loc[df.t == 95.5, col] == 1).reset_index()[col], "response"
            ] = col

        for x in ["init_w", "eps", "mu", "s", "seed"]:
            vals[x] = eval(x)
        res = pd.concat([res, vals])


res["Hpersnonfoc"] = res.Hpers - res.Hpersfoc
metric_cols[1] = "Hpersnonfoc"
res = res.drop(columns=["Hpers"])


# %%

# ----------------------------------------------
# -------    PRINT TABLE
# ----------------------------------------------

pressures = [4]
for s_ext in pressures:
    print(
        "".join(["#"] * 50)
        + f"\n time = 91-100\n"
        + "".join(["#"] * 50)
        + f"\n s_ext = {s_ext}\n"
        + "".join(["#"] * 50)
    )
    print(" & " + " & ".join([""] + negcols) + " & \\\\ \\hline")
    metrics_table = [
        "x_focal",
        "extr_nonfoc",
        "n_nbs",
        "absOm_tot",
        "absOm_foc",
        "tb_tot",
        "tb_foc",
        "clust",
        "bc",
        "Hpersfoc",
        "Hpersnonfoc",
        "Hsoc",
    ]
    for metric in metrics_table:
        print(f"{metric2titleVerb[metric]} &  " + f"{metric2title[metric]} &  ", end="")
        for r in negcols:
            a = res.loc[(res.response == r) & (res.s == s_ext), metric]  # .mean()
            if len(a) > 0:
                print(f"${a.mean():.2f} \\pm {a.std():.2f}$", end=" & ")
            else:
                print(" ", end=" & ")
        print("\\\\")

        if metric == metrics_table[-1]:
            print(f"proportion &  & ", end="")
            props = res.loc[res.s == s_ext, "response"].value_counts() / (
                len(res.id.unique()) * len(res.seed.unique())
            )
            summeNeg = (
                props[negcols].sum()
                if "resistant" in props
                else props[["compliant", "resilient"]].sum()
            )
            for r in negcols:
                if r in props:
                    print(
                        f"${100*props[r]/summeNeg:.1f}\,\%$",
                        end=" & ",
                    )
                else:
                    print(f"${0.0:.1f}\,\%$ & ")
            print("\\\\")


# %%

# ----------------------------------------------
# -------    Exploration
# ----------------------------------------------
# res.query("s==1 and eps==1").groupby("response")[metric_cols].mean().loc[['compliant',
#  'resilient',
#  'resistant']].T
# %%


# plt.rcParams.update({"font.size": 10})
# bigfs = 16
# smallfs = 15
# plt.rcParams.update({"font.size": bigfs})
# plt.rcParams.update({"axes.titlesize": bigfs})
# plt.rcParams.update({"axes.labelsize": bigfs})
# plt.rcParams.update({"legend.fontsize": smallfs})
# plt.rcParams.update({"xtick.labelsize": smallfs})
# plt.rcParams.update({"ytick.labelsize": smallfs})
# s4 = res.query("s==4 and eps==1")
# sns.pairplot(
#     s4.loc[
#         s4.response.isin(["compliant", "resilient", "resistant"]),
#         metric_cols2 + ["response"],
#     ],
#     hue="response",
#     palette=cmap,
# )

# %%

# ----------------------------------------------
# -------    COHENS D
# ----------------------------------------------


def cohens_d(a, b):
    var_a = a.var()
    var_b = b.var()
    pooled_var = (var_a + var_b) / 2
    # Treat near-zero variance as zero
    pooled_var[pooled_var < 1e-10] = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (a.mean() - b.mean()) / pooled_var**0.5
    return d.where(pooled_var > 0, other=np.nan)


res["noncompliant"] = np.nan
res.loc[res.response.isin(["resistant", "resilient"]), "noncompliant"] = 1
res.loc[res.response.isin(["compliant"]), "noncompliant"] = 0

pressures = [1, 2, 4, 8, 16]
cds_df = []
for s in pressures:
    res_s = res.query(f"s=={s} and eps=={eps}")
    allagents = res_s.dropna(subset=["noncompliant"])
    for ty in ["rR-C", "R-r"]:
        if ty == "rR-C":
            group = allagents.loc[allagents["noncompliant"] == 1, metric_cols2]
            base = allagents.loc[allagents["noncompliant"] == 0, metric_cols2]
        elif ty == "R-r":
            group = allagents.loc[allagents.response == "resistant", metric_cols2]
            base = allagents.loc[allagents.response == "resilient", metric_cols2]
        cds = cohens_d(group, base)
        for metric in metric_cols2:
            cds_df.append(
                [s, metric2title[metric], metric2titleVerb[metric], ty, cds.loc[metric]]
            )
cohensd_df = pd.DataFrame(cds_df, columns=["s", "metricMath", "metric", "type", "d"])

# %%
# ----------------------------------------------
# -------    REGRESSION
# ----------------------------------------------
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

dependent_vars = [
    "n_nbs",
    # "tb_tot",
    "tb_foc",
    # "absOm_tot",
    "absOm_foc",
    "clust",
    # 'bc_focal',
    "x_focal",
    # 'extr_nonfoc'
]
control_vars = []
coef_df = pd.DataFrame()
for s in pressures:
    for ty in ["rR-C", "R-r"]:
        df = res.query(f"s=={s} and eps=={eps}")
        label_map = dict(zip(response_cols, [np.nan] * 6))
        if ty == "R-r":
            label_map["resistant"] = 1
            label_map["resilient"] = 0
        elif ty == "rR-C":
            label_map["resistant"] = 1
            label_map["resilient"] = 1
            label_map["compliant"] = 0
            label_map["latecompliant"] = 0

        df["response_group"] = df["response"].map(label_map)
        df = df.dropna(subset=["response_group"])

        X = df[dependent_vars + control_vars]
        scaler = StandardScaler()
        X_scaled_df = pd.DataFrame(
            scaler.fit_transform(X), columns=dependent_vars + control_vars
        )
        y = df["response_group"].values
        X_scaled_df = X_scaled_df.loc[:, X.std() > 1e-6]

        model = sm.Logit(y, sm.add_constant(X_scaled_df)).fit()

        coefs = pd.DataFrame(
            {
                "coefficient": model.params[1:],  # exclude intercept
                "odds_ratio": np.exp(model.params[1:]),
                "pvalue": model.pvalues[1:],
                "ci_low": model.conf_int()[0][1:],
                "ci_high": model.conf_int()[1][1:],
            },
            index=dependent_vars + control_vars,
        )
        coefs["type"] = ty
        coefs["s"] = s

        print(f"\n{ty} --- Intercept: {model.params["const"]:.4f}")
        print(f"{ty} --- Pseudo R²:  {model.prsquared:.4f}")  # better than accuracy

        coef_df = pd.concat([coef_df, coefs])

coef_df["metric"] = coef_df.index.map(metric2titleVerb)


combined = (
    cohensd_df.set_index(["type", "metric", "s"])
    .join(coef_df.set_index(["type", "metric", "s"])["coefficient"])
    .reset_index()
)
# ----------------------------------------------
# -------    PLOTTING
# ----------------------------------------------

# %%
showRegression = True
diss = False
nmetric = len(combined.metric.unique())
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(18 / 2.54, 8 / 2.54))
# axs[0].scatter([],[],marker="o", c="grey", s=5, edgecolor="grey", label="regression\ncoefficient")
if showRegression:
    dotpatch = mpl.lines.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="lightgrey",
        markeredgecolor="grey",
        markeredgewidth=0.7,
        markersize=4,
        label="regression\ncoefficient",
    )
patch = mpatches.Patch(color="grey", label=r"Cohen's $d$")
axs[0].legend(
    handles=[patch] if not showRegression else [patch, dotpatch],
    fontsize=smallfs,
    borderpad=0.25,
    handlelength=1,
    handleheight=0.5,
    handletextpad=0.7,
    facecolor="gainsboro",
    loc="lower right",
    edgecolor="none",
    framealpha=0.5,
)
for n, ty in enumerate(["rR-C", "R-r"]):
    ax = axs[n]
    ss = [1, 2, 4] if ty == "R-r" else [1, 2, 4, 8, 16]
    plot_df = combined.loc[(cohensd_df.type == ty) & (combined.s.isin(ss))].sort_values(
        "s", ascending=False
    )
    metric_order = list(metric2titleVerb.values())[::-1]
    sns.barplot(
        plot_df,
        x="d",
        y="metric",
        hue="s",
        hue_order=pressures[::-1],
        dodge=True,
        ax=ax,
        palette=cmapS,
        legend=False,
        zorder=10,
        order=metric_order,
        errorbar=None,
        alpha=0.8,
        lw=0,
    )
    if showRegression:
        ax2 = ax.twiny()
        if metric2titleVerb["x_focal"] in plot_df.metric.values:
            print("control for x_focal")
            plot_df = plot_df.loc[
                ~(plot_df.metric == metric2titleVerb["x_focal"])
            ].copy()
        sns.stripplot(
            data=plot_df.dropna(subset=["coefficient"]),
            x="coefficient",
            y="metric",
            hue="s",
            hue_order=pressures[::-1],
            order=metric_order,
            dodge=True,
            jitter=False,
            palette=cmapS,
            ax=ax2,
            legend=False,
            size=3,
            zorder=20,
            edgecolor="k",
            linewidth=0.4,
        )
        ax2.set_xlabel("Regression Coefficient (dots)", fontsize=smallfs)
        if diss:
            ax2.set_xlim(-1, 1)
        else:
            ax2.set_xlim(-3, 3)
    ax.set_xlabel(r"Cohen's $d$ (bars)", fontsize=smallfs)
    ax.set_title(
        (
            "\nResistant/Resilient vs. Compliant"
            if ty == "rR-C"
            else "\nResistant vs. Resilient"
        ),
        fontsize=bigfs + 1,
    )
    ax.set_ylabel("")
    maxy = (
        np.argsort(
            [p.get_y() for p in ax.patches],
        )[-10:-5]
        if n == 0
        else np.argsort(
            [p.get_y() for p in ax.patches],
        )[-6:-3]
    )
    for kk, (barind, text) in enumerate(zip(maxy, [rf"$s={s}$" for s in ss[::-1]])):
        bar = ax.patches[barind]
        h = bar.get_width()
        if np.isnan(h):
            continue

        y = bar.get_y() + bar.get_height() / 2
        x = bar.get_width()
        bar_fc = bar.get_facecolor()
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(
                1.2,
                y
                + (1.3 if n == 0 else 1)
                - (0.56 if n == 0 else 0.36) * (len(maxy) - kk),
            ),
            ha="left",
            va="center",
            rotation=0,
            fontsize=smallfs - 1,
            arrowprops=dict(
                arrowstyle="-",
                lw=1,
                connectionstyle="arc,rad=0.1",  # vertical then tilt
                color=bar_fc,
            ),
            bbox={"pad": 0.02, "fc": "white", "ec": "none"},
            color=bar_fc,
            zorder=20,
        )

    ax.vlines([0], -0.5, nmetric + 0.5, lw=0.5, color="grey")
    xlim = 2
    ax.hlines([0.5 + k for k in range(nmetric)], -xlim, xlim, lw=0.5, color="grey")
    ax.hlines(
        [nmetric - 2.5, nmetric - 3.5, nmetric - 9.5], -xlim, xlim, lw=1.5, color="grey"
    )
    ax.set_ylim(-0.5, nmetric - 0.5)
    ax.set_xlim(-xlim, xlim)
    if showRegression:
        ax2.set_ylim(ax.get_ylim())

# draw once so tick labels exist
fig.canvas.draw()
if showRegression:
    for label in axs[0].get_yticklabels():
        if label.get_text() in list(coef_df.metric.unique()):
            label.set_fontweight("bold")


fig.subplots_adjust(
    left=0.2, top=0.82 if showRegression else 0.92, right=0.98, bottom=0.12
)
fname = f"2026-04_figs/fig4_mu{mu}{
names[(init_w,eps,mu,fixedBNat100)] if eps!=1 else ''}"

print(fname)
plt.savefig(fname + ".png", dpi=600)
plt.savefig(fname + ".pdf")
# baseline{'staticBNafterBurnIn' if staticBNafterBurnIn else ''}{'_withDiss' if showRegression and diss else ''}

# %%


display(combined)
# %%
