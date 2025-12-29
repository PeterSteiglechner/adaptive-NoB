# %%
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import netCDF4
import json
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

ds = xr.load_dataset("processed_data/2025-12-29_modelAdaptiveBN_results.ncdf", engine="netcdf4")

# %%
belief_dimensions = list(range(ds.attrs["M"]))
edgeNames = combinations(belief_dimensions, 2)
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
response_map_inv = {val: key for key, val in response_map.items()}
cmap = dict(
    zip(
        responses+["NA"],
        ["#7fc97f", "#fdc086", "#386cb0", "#beaed4", "#f0027f", "#bf5b17", "#666666"],
    )
)


# %%
countResponses = []
for adaptive in [False, True]:
    for s_ext in ds.s_ext.values:
        if s_ext == 0:  # skip no intervention
            continue
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
dff["s_ext"] = dff["s_ext"].astype(float)
dff["normalized_count"] = dff["count"] / (len(ds.seed) * len(ds.agent_id))
dff["response"] = dff["response"].replace("NA", np.nan)
df = dff.dropna().copy()
df["s_ext_log2"] = np.log2(df["s_ext"])
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12 / 2.54, 8 / 2.54))
for ax, adaptive in zip(axs, [False, True]):
    subset = df.loc[df.adaptive == adaptive]
    grouped = (
        subset.groupby(["s_ext_log2", "response"])["normalized_count"].sum().reset_index()
    )
    pivoted = grouped.pivot(
        index="s_ext_log2",
        columns="response",
        values="normalized_count",
    ).fillna(0)[responses]
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
# ----------------------------------------------
# -----    DESCRIBE METRICS STATS    ------
# ----------------------------------------------

belief_metrics = [ "focal_belief", "abs_meanbelief_nonfocal"]
structure_metrics = ["bn_alpha",   "bn_avg_weighted_clustering", "bn_betweenness_centrality", "bn_abs_meanedge_tot", "bn_abs_meanedge_focal"]
structureValue_metrics = ["nr_balanced_triangles_tot", "nr_balanced_triangles_focal",  "bn_expected_influence",  ]
social_net_metrics = ["n_neighbours"] 
energy_metrics = [ "energy", "personalBN_energy", "personalBN_focal_energy", "personalBN_nonfocal_energy", "external_energy","social_energy",]
all_metrics = belief_metrics + social_net_metrics + energy_metrics + structureValue_metrics + structure_metrics
t = 94.5
ttt=  f"{t-4.5}-{t+4.5}" if (t%10) == 4.5 else t
s_ext = 4
print("".join(["#"]*50)+f"\n time = {ttt}\n"+"".join(["#"]*50)+f"\n s_ext = {s_ext}\n"+"".join(["#"]*50))
ds["energy"] = ds["personalBN_energy"] + ds["social_energy"] + ds["external_energy"]
ds["personalBN_nonfocal_energy"] = ds["personalBN_energy"] - ds["personalBN_focal_energy"] + ds["external_energy"]
for metric in all_metrics:
    print("".join(["#"]*3)+f" {metric}")

    a = (
        ds
        .sel(adaptive=True, time=t, s_ext=s_ext)[[metric, "response_type"]]
        .to_dataframe()
        .reset_index()
        .groupby("response_type")[metric]
        .agg(mean="mean", sd="std", count="count")
        .rename(columns={"mean":metric})
        .reset_index()
    )
    a["response_type"] = a["response_type"].map(response_map_inv)
    #display(a)
    for r in ["persistent-positive", "compliant", "resilient", "resistant"]:
        print(f"{r}: {a.loc[a.response_type==r, metric].values[0]:.3f} +- {a.loc[a.response_type==r, "sd"].values[0]:.3f}")
print("".join(["#"]*3)+f" COUNT")
for r in a.response_type.unique():
    print(f"{r}: {a.loc[a.response_type==r, "count"].values[0]:d} ")


# %%
# ----------------------------------------------
# -------    PLOT METRICS
# ----------------------------------------------

t=294.5
cmap["NA"]="grey"
for metric in all_metrics:
    print("".join(["#"]*10)+f".  {metric}")
    fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 7 / 2.54), sharey=True)
    for idx, adaptive_val in enumerate([False, True]):
        data_subset = ds.sel(time=t, adaptive=adaptive_val, s_ext=[1,2,4,8,16])[["response_type", metric]].to_dataframe().reset_index()[["s_ext", "response_type", metric]]
        data_subset["response_type"] = data_subset["response_type"].map(response_map_inv)
        sns.boxplot(
            data=data_subset,
            x="s_ext",
            y=metric,
            hue="response_type",
            ax=axes[idx],
            fliersize=0,
            palette=cmap,
            legend=adaptive_val == False,
            boxprops={"edgecolor": "white", "lw":0 }
        )
        if adaptive_val == False:
            h,l = axes[idx].get_legend_handles_labels()
            axes[idx].legend(h,l,title="",ncol=2,fontsize=7, loc="lower left", columnspacing=0.2, handletextpad=0.1, handlelength=1)
        sns.stripplot(
            data=data_subset,
            x="s_ext",
            y=metric,
            hue="response_type",
            ax=axes[idx],
            size=1,
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
        axes[idx].set_ylabel(metric)
    ttt=  f"{t-4.5}-{t+4.5}" if (t%10) == 4.5 else t
    name=metric+f" t={ttt}"
    fig.suptitle(name, y=1, va="top", fontsize=bigfs)
    plt.savefig(resultsfolder + f"figs/characteristics_t{ttt}_{metric}.png")

# %%
# ----------------------------------------------
# -------    PLOT ENERGIES OVER TIME
# ----------------------------------------------
s_ext =4
fig, axs = plt.subplots(2,3, sharex=True, sharey=True)
for ax, metric in zip(axs.flatten(), energy_metrics):
    a = (
        ds
        .sel(adaptive=True, time=[4.5,94.5,144.5,194.5,294.5], s_ext=s_ext)[[metric, "response_type"]]
        .to_dataframe()
        .reset_index()
        .groupby(["response_type", "time"])[metric]
        .agg(mean="mean", sd="std", count="count")
        .rename(columns={"mean":metric})
        .reset_index()
    )
    a["response_type"] = a["response_type"].map(response_map_inv)
    sns.lineplot(a, ax=ax, x="time", y=metric, hue="response_type", palette=cmap, marker="o", legend=metric=="social_energy", errorbar=None)
    for (_, g), line in zip(a.groupby("response_type"), ax.lines):
        ax.errorbar(
            g["time"],
            g[metric],
            yerr=g["sd"],
            fmt="none",
            capsize=3,
            color=cmap[g.response_type.iloc[0]],
            linewidth=1
        )
    if metric=="social_energy":
        leg = ax.get_legend()
        leg.set_title("")
        leg.set_loc("lower right")
    ax.set_title(metric)
    ax.set_ylabel("energy")
    if metric=="external_energy":
        ax.text(125, ax.get_ylim()[0]+8, "external\npressure\n"+rf"$s={s_ext}$", va="center", ha="center")

yl,yh = ax.get_ylim()
for ax in axs.flatten():
    ax.fill_between([100,150], [yl,yl], [yh,yh], color="red", alpha=(1+np.log2(s_ext))/5*0.4, zorder=-1, lw=0)
fig.subplots_adjust(left=0.09, top=0.93, right=0.98, bottom=0.1)
plt.savefig(resultsfolder + f"figs/energiesOverTime_s{s_ext}.png")

# %%
# %%
# ----------------------------------------------
# -------    PLOT FOCAL BELIEFS OVER TIME
# ----------------------------------------------
metric = "focal_belief"
fig, axs = plt.subplots(2,3, sharex=True, sharey=True)
for ax, s in zip(axs.flatten(), [0,1,2,4,8,16]):
    a = (
            ds
            .sel(adaptive=True, time=[4.5,94.5,144.5,194.5,294.5], s_ext=s)[[metric, "response_type"]]
            .to_dataframe()
            .reset_index()
            .groupby(["response_type", "time"])[metric]
            .agg(mean="mean", sd="std", count="count")
            .rename(columns={"mean":metric})
            .reset_index()
    )
    a["response_type"] = a["response_type"].map(response_map_inv)
    sns.lineplot(a, ax=ax, x="time", y=metric, hue="response_type", palette=cmap, marker="o", legend=False, errorbar=None)
    for (_, g), line in zip(a.groupby("response_type"), ax.lines):
        ax.errorbar(
            g["time"],
            g[metric],
            yerr=g["sd"],
            fmt="none",
            capsize=3,
            color=cmap[g.response_type.iloc[0]],
            linewidth=1
        )
    ax.set_title(fr"$s={s}$")
    ax.set_ylabel("mean focal belief")
    # if s==1:
    #     leg = ax.get_legend()
    #     leg.set_title("")
    #     leg.set_loc("lower right")

    ax.set_ylim(-1.15,1.15)
yl,yh = ax.get_ylim()
for ax, s in zip(axs.flatten(),[0,1,2,4,8,16]):
    if s>0:
        ax.fill_between([100,150], [yl,yl], [yh,yh], color="red", alpha=(1+np.log2(s))/5*0.4, zorder=-1, lw=0)
fig.subplots_adjust(left=0.07, top=0.93, right=0.98, bottom=0.1)
plt.savefig(resultsfolder + f"figs/focalBeliefsOverTime.png")

# %%
