# %%
import numpy as np
import pandas as pd
from itertools import combinations
import string
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import json
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import xarray as xr

plt.rcParams.update({"font.size": 10})
bigfs = 10
smallfs = 8
plt.rcParams.update({"font.size": bigfs})
plt.rcParams.update({"axes.titlesize": bigfs})
plt.rcParams.update({"axes.labelsize": bigfs})
plt.rcParams.update({"legend.fontsize": smallfs})
plt.rcParams.update({"xtick.labelsize": smallfs})
plt.rcParams.update({"ytick.labelsize": smallfs})


# %%
resultsfolder = "2025-12-16/"

ds = xr.load_dataset("processed_data/2025-12-16_modelAdaptiveBN_results.ncdf")

#################################
#####  #################################
#####  #################################
#####  VISUALISE   #####
#################################   #####
#################################   #####
#################################


# %%
belief_dimensions = list(ds.belief.values)
edgeNames = list(ds.edge.values)
focal_dim = ds.attrs["focal_dim"]
eval_time = ds.attrs["evaluation_time_for_response"]
responses = [
    "persistent-positive",
    "non-persistent-positive",
    "compliant",
    "late-compliant",
    "resilient",
    "resistant",
]
response_map = {r: n for n, r in enumerate(responses)}
response_map["NA"] = 99
cat_type = pd.CategoricalDtype(categories=responses, ordered=True)
svals = [0, 16]

cmap = dict(
    zip(
        responses,
        ["#7fc97f", "#fdc086", "#386cb0", "#beaed4", "#f0027f", "#bf5b17", "#666666"],
    )
)


# %%
countResponses = []
for adaptive in [False, True]:
    for s_ext in ds.s_ext.values:
        if s_ext == 0:  # skip no intervention
            continue

        # Get response types for this condition
        resp = ds.response_type.sel(adaptive=adaptive, s_ext=s_ext)

        for response_type, r in response_map.items():
            count = (resp == r).sum().values
            countResponses.append(
                {
                    "adaptive": adaptive,
                    "s_ext": s_ext,
                    "response": response_type,
                    "count": int(count),
                }
            )

dff = pd.DataFrame(countResponses)
dff = dff.sort_values(["adaptive", "s_ext", "response"])

dff["normalized_count"] = dff["count"] / (len(ds.seed) * len(ds.agent_id))
dff["response"] = dff["response"].replace("NA", np.nan)
df = dff.dropna()
df["s_ext"] = np.log2(df["s_ext"])
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12 / 2.54, 8 / 2.54))
for ax, adaptive in zip(axs, [False, True]):
    subset = df.loc[df.adaptive == adaptive]
    subset["s_ext"] = subset["s_ext"].astype(float)
    # subset["response"] = subset.response.map(response_map)
    grouped = (
        subset.groupby(["s_ext", "response"])["normalized_count"].sum().reset_index()
    )
    pivoted = grouped.pivot(
        index="s_ext",
        columns="response",
        values="normalized_count",
    ).fillna(0)[responses]
    # pivoted = pivoted.div(pivoted.sum(axis=1), axis=0)
    ax = pivoted.plot(
        kind="bar",
        stacked=True,
        color=[cmap[r] for r in pivoted.columns],
        alpha=0.8,
        width=0.7,
        legend=False,
        ax=ax,
    )
    ax.set_ylabel("Proportion")
    ax.set_title("adaptive belief networks" if adaptive else "fixed belief networks")
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(
        [f"$2^{int(float(v.get_text())):0d}$" for v in ax.get_xticklabels()], rotation=0
    )
    ax.set_xlabel(r"external pressure $s$")
    ax.set_ylim(-0.0, 1.0)
for (x, y), type in zip(
    [(0.6, 0.1), (0.2, 0.4), (2.6, 0.7), (0.3, 0.56), (0.3, 0.64), (0.0, 0.9)],
    responses,
):
    bboxprops = dict(
        boxstyle="round",
        facecolor=cmap[type],
        edgecolor="white",
        alpha=1,
    )
    axs[0].text(
        x,
        y,
        type,
        color=(
            "white"
            if type in ["resilient", "resistant", "compliant", "late-compliant"]
            else "k"
        ),
        va="center",
        ha="left",
        bbox=bboxprops,
        fontsize=8,
    )
fig.subplots_adjust(left=0.1, right=0.99, top=0.9, bottom=0.15, hspace=0.05)
# fig.set_facecolor("pink")
plt.savefig(
    resultsfolder + f"figs/results_proportions_evaluatedAtT={eval_time}.png", dpi=300
)
# %%

# %%
time = 194.5
ttt = (190, 199)
times = f"{ttt[0]}–{ttt[1]}"
nonfocalbeliefs = [f"{bel}" for bel in belief_dimensions if not bel == focal_dim]
focaledges = [f"{edge}" for edge in edgeNames if focal_dim in edge]
nonfocaledges = [f"{edge}" for edge in edgeNames if not focal_dim in edge]
for foc in ["focal", "nonfocal"]:
    for what in ["edge_weight", "belief_value"]:
        for absVal in ["abs", ""]:
            if foc == "focal" and what == "belief_value":
                variableName = f"Focal Beliefs at $t={times}$"
                var = f"{foc}_{absVal}_{what}"
                currdf = (
                    ds[[what, "response_type"]]
                    .sel(belief=focal_dim)
                    .sel(time=time)
                    .to_dataframe()
                    .reset_index()
                    .drop(columns=["belief", "agent_id"])
                )
            elif foc == "nonfocal" and what == "belief_value":
                variableName = f"Non-Focal Mean {'Abs ' if "abs" in absVal else ''}Beliefs at $t={ttt}$"
                var = f"mean_{foc}_{absVal}_{what}"
                currdf = (
                    ds[[what, "response_type"]]
                    .sel(belief=ds.belief.isin(nonfocalbeliefs))
                    .sel(time=time)
                    .to_dataframe()
                    .reset_index()
                    # .drop(columns=["agent_id"])
                )

            elif foc == "focal" and what == "edge_weight":
                variableName = f"Focal Mean {'Abs ' if "abs" in absVal else ''}Edge Weights at $t={ttt}$"
                var = f"mean_{foc}_{absVal}_{what}"
                currdf = (
                    ds[[what, "response_type"]]
                    .sel(edge=ds.edge.isin(focaledges))
                    .sel(time=time)
                    .to_dataframe()
                    .reset_index()
                    # .drop(columns=["agent_id"])
                )

            elif foc == "nonfocal" and what == "edge_weight":
                variableName = f"Non-Focal Mean {'Abs ' if "abs" in absVal else ''}Edge Weights at $t={ttt}$"
                var = f"mean_{foc}_{absVal}_{what}"
                # variables =  ["belief"]
                currdf = (
                    ds[[what, "response_type"]]
                    .sel(edge=ds.edge.isin(nonfocaledges))
                    .sel(time=time)
                    .to_dataframe()
                    .reset_index()
                    # .drop(columns=["agent_id"])
                )
            # vars_to_keep = ["response_type"] + [what.split("_")[0]]
            coords_to_keep = ["response_type", "adaptive", "seed", "s_ext", "agent_id"]
            # dff = ds[vars_to_keep].to_dataframe().reset_index()
            # dff = dff[coords_to_keep + vars_to_keep]
            if "abs" in var.lower():
                currdf[what] = currdf[what].abs()
            if "mean" in var:
                currdf = (
                    currdf.pivot_table(
                        values=what, columns=what.split("_")[0], index=coords_to_keep
                    )
                    .mean(axis=1)
                    .reset_index(name=var)
                )
                # currdf[var] = currdf[variables].mean(axis=1)
            else:
                currdf[var] = currdf[what]

            response_map_inv = {val: key for key, val in response_map.items()}
            currdf["response_type"] = (
                currdf["response_type"].map(response_map_inv).astype(cat_type)
            )

            plot_data = currdf  # []

            fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 7 / 2.54), sharey=True)

            for idx, adaptive_val in enumerate([False, True]):
                data_subset = plot_data[plot_data["adaptive"] == adaptive_val]

                sns.boxplot(
                    data=data_subset,
                    x="s_ext",
                    y=var,
                    hue="response_type",
                    ax=axes[idx],
                    fliersize=0,
                    palette=cmap,
                    legend=adaptive_val == False,
                )
                if adaptive_val == False:
                    leg = axes[idx].get_legend()
                    # leg.set_title("")
                sns.stripplot(
                    data=data_subset,
                    x="s_ext",
                    y=var,
                    hue="response_type",
                    ax=axes[idx],
                    size=0.5,
                    alpha=0.4,
                    dodge=True,
                    palette=cmap,
                    legend=False,
                )
                axes[idx].set_title(
                    f'{"Adaptive" if adaptive_val else "Fixed"}',
                    x=0.8 if adaptive_val else 0.2,
                )
                axes[idx].set_xlabel("external pressure $s$")
                axes[idx].set_ylabel(var)
            fig.suptitle(variableName, y=1, va="top", fontsize=bigfs)
            plt.savefig(resultsfolder + f"figs/characteristics_t{ttt}_{var}.png")

# %%

# %%

#################################
#####  TIME SERIES   #####
#################################
