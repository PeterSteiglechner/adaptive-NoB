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
bigfs = 9
smallfs = 7
plt.rcParams.update({"font.size": bigfs})
plt.rcParams.update({"axes.titlesize": bigfs})
plt.rcParams.update({"axes.labelsize": bigfs})
plt.rcParams.update({"legend.fontsize": smallfs})
plt.rcParams.update({"xtick.labelsize": smallfs})
plt.rcParams.update({"ytick.labelsize": smallfs})


# %%
condition_string = "baselineConfig"
ds = xr.load_dataset(f"processed_data/2025-12-29_modelAdaptiveBN_{condition_string}_results_metricsOnly.ncdf", engine="netcdf4")

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

#%%
# ----------------------------------------------
# -------    NO PRESSURE: VARIANCE AND CONSENSUS
# ----------------------------------------------
adaptive=False
a = (ds.sel(adaptive=adaptive, s_ext=0, time=94.5)[
        "focal_belief"
    ]>=0).sum(dim="agent_id")
(a>=98).sum(dim="seed"), (a<=2).sum(dim="seed") 

ds.sel(adaptive=adaptive, s_ext=0, time=94.5)[
        "focal_belief"
    ].std(dim="agent_id").mean(dim="seed")


#%%
# ----------------------------------------------
# -------    RESPONSE TYPES: FREQUENCY
# ----------------------------------------------

examplesim = pd.read_csv("sims/adaptiveBN_M-10-randomInitialOps_n-100-(p=0.1)_eps1.0-m1_lam0.005_rho0.33_beta3.0_initialW-0.2_ext-100-149-on-0-strength4_seed98_detailed.csv")
examplesim.loc[examplesim.time.isin([99,149,199])].pivot_table(values="0", index="time", columns="agent_id")
T = 200
dim = "0"
adaptive = True
ext_pressure_strength = 4
seed = 0  # np.random.randint(10)
window = 5
df = examplesim[["0", "agent_id", "time"]]
df = df.sort_values(["agent_id", "time"])
t = 200
final_values = df.loc[df.time == t, ["agent_id", dim]]
if window > 0:
    df["belief_smooth"] = df.groupby("agent_id")[dim].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
else:
    df["belief_smooth"] = df[dim]
df_pivot = df.pivot(index="time", columns="agent_id", values="belief_smooth")


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
fig, axs = plt.subplots(1,4, gridspec_kw={"width_ratios":[1.8,0.15, 1,1]}, figsize=(16 / 2.54, 6 / 2.54))
axs[1].axis("off")
axs[3].sharex(axs[2])
for ax, adaptive in zip(axs[2:], [False, True]):
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
    ax.set_ylabel("proportion")
    ax.set_title("adaptive belief networks" if adaptive else "fixed belief networks", fontsize=bigfs)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(
        [f"$2^{int(float(v.get_text())):0d}$" for v in ax.get_xticklabels()], rotation=0
    )
    ax.set_xlabel("")
    ax.set_ylim(-0.0, 1.0)
    ax.set_yticks([-0.,0.2,0.4,0.6,0.8,1.0])
    if ax==axs[3]:
        ax.set_yticklabels([])
        ax.set_ylabel("")
axs[2].set_xlabel(r"external pressure $s$", x=1)
# for (x, y), type in zip(
#     [(0.6, 0.15), (0.15, 0.35), (2.6, 0.7), (0.3, 0.55), (0.3, 0.66), (0.0, 0.9)],
#     responses,
# ):
#     bboxprops = dict(
#         boxstyle="round",
#         facecolor=cmap[type],
#         edgecolor="white",
#         alpha=1,
#     )
#     axs[1].text(
#         x,
#         y,
#         type,
#         color=(
#             "white"
#             if type in ["resilient", "resistant", "compliant", "late-compliant"]
#             else "k"
#         ),
#         va="center",
#         ha="left",
#         bbox=bboxprops,
#         fontsize=8,
#     )
bboxprops = dict(
        boxstyle="round",
        facecolor=cmap["late-compliant"],
        edgecolor=None,
        alpha=0.6,
    )
ap = dict(arrowstyle="-", connectionstyle="arc3,rad=0", color="black", shrinkA=0, shrinkB=0,)
axs[3].annotate("late-compliant", (0.3,0.68), (0.8,0.68), fontsize=smallfs, arrowprops=ap,bbox=bboxprops)
bboxprops["facecolor"] = cmap["non-persistent-positive"]
axs[3].annotate("non-persistent-\npositive", (0.3,0.4), (0.8,0.2), fontsize=smallfs, arrowprops=ap,bbox=bboxprops)

ax_main = axs[0]
resistant = 50 # resistant
persistentpositive = 4
resilient = 55
compliant = 67
ax_main.plot([],[],lw=0.7, alpha=0.4, label="agent", color="grey")
leg= ax_main.legend(loc="center left")
df_pivot = df_pivot.loc[df_pivot.index <= t]
df_pivot.plot(ax=ax_main, lw=0.5, alpha=0.4, legend=False, color="grey", label="_")
for i, name in zip([resistant, resilient, compliant, persistentpositive],["resistant", "resilient", "compliant", "persistent-positive"]):
    df_pivot.T.loc[i].plot(ax=ax_main, lw=2, ls="-", color=cmap[name], alpha=0.8, legend=False, label="_", )
ax_main.set_xlabel("time", fontsize=bigfs)
ax_main.set_ylabel("focal belief", fontsize=bigfs, va="top")
ax_main.set_xlim(0, T)
if len(dim) == 1:
    ax_main.set_ylim(-1, 1)
ax_main.set_yticks([-1,0,1])
ax_main.set_clip_on(False)
leg = ax_main.get_legend()
leg.set_bbox_to_anchor((0.05,0.2,0.3,0.3))
if ext_pressure_strength > 0 and t > 100:
    # ["#640000", "#850000", "#B20000", "#DE0000", "#FF0000"]
    int_colors = dict(zip([1, 2, 4, 8, 16], [0.1, 0.175, 0.25, 0.325, 0.4]))
    y0, y1 = ax_main.get_ylim()
    xx = [100, 150]
    xx = [ttt for ttt in xx if ttt <= t]
    if len(xx) > 0:
        ax_main.fill_between(
            xx,
            [y0] * len(xx),
            [y1] * len(xx),
            color="red",
            alpha=int_colors[ext_pressure_strength],
            zorder=-1,
            lw=0,
        )


for (x, y), type in zip(
    [(20, 0.7), (110,-0.4),(60,0.3), (162, -0.4)],
    ["persistent-positive", "resistant", "compliant", "resilient"],
):
    bboxprops = dict(
        boxstyle="round",
        facecolor=cmap[type],
        edgecolor="white",
        alpha=0.8,
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
        fontsize=smallfs,
    )

axs[0].text(125,0.36, "external\npressure", ha="center", fontsize=smallfs)
axs[0].text(0.02,1.02,"example simulation (adaptive, $s=$"+f"{ext_pressure_strength}, smoothed)", transform=ax_main.transAxes, ha="left", va="bottom", fontsize=smallfs)
# fig.set_facecolor("pink")
import string
for n, ax in enumerate(axs[[0,2,3]]): 
    ax.text(0.025, 0.975, string.ascii_uppercase[n], fontsize=12, fontdict={"weight":"bold"},va="top", ha="left", transform=ax.transAxes)
fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.17, wspace=0.1)

plt.savefig(
     f"figs/results_{condition_string}_proportions_evaluatedAtT={eval_time}.png", dpi=300
)



# %%
# ----------------------------------------------
# -----    DESCRIBE METRICS STATS    ------
# ----------------------------------------------
metric2title = dict(
    focal_belief = r"$x_{foc}$",
    abs_meanbelief_nonfocal = r"$|X_{non\text{-}foc}|$",
    bn_expected_influence = r"$\langle\delta x_{foc}\rangle$", 
    n_neighbours = r"$|\mathcal{K}|$", 
    nr_balanced_triangles_tot = r"BN-$\alpha$",
    nr_balanced_triangles_focal = r"BN-$\alpha_{foc}$",
    bn_abs_meanedge_tot = r"BN-$|\Omega|$",
    bn_abs_meanedge_focal = r"BN-$|\Omega_{foc}|$",
    bn_avg_weighted_clustering = r"BN-$CC$",
    bn_betweenness_centrality = r"BN-$BC$",
    personalBN_nonfocal_energy = r"$D_{non\text{-}foc}$",
    personalBN_focal_energy = r"$D_{BN\text{-}foc}$",
    social_energy = r"$D_{social}$",
    external_energy = r"$D_{ext}$",    
    energy = r"$D_{tot}$",
)
# belief_metrics = [ "focal_belief", "abs_meanbelief_nonfocal"]
# structure_metrics = ["bn_alpha",   "bn_avg_weighted_clustering", "bn_betweenness_centrality", "bn_abs_meanedge_tot", "bn_abs_meanedge_focal"]
# structureValue_metrics = ["nr_balanced_triangles_tot", "nr_balanced_triangles_focal",  "bn_expected_influence",  ]
# social_net_metrics = ["n_neighbours"] 
# energy_metrics = [ "energy", "personalBN_energy", "personalBN_focal_energy", "personalBN_nonfocal_energy", "external_energy","social_energy",]
all_metrics = metric2title.keys()#belief_metrics + social_net_metrics + energy_metrics + structureValue_metrics + structure_metrics
t = 94.5
ttt=  f"{t-4.5}-{t+4.5}" if (t%10) == 4.5 else t
s_ext = 4
print("".join(["#"]*50)+f"\n time = {ttt}\n"+"".join(["#"]*50)+f"\n s_ext = {s_ext}\n"+"".join(["#"]*50))
print(" & ".join(["", "compliant", "resilient", "resistant"]))
ds["energy"] = ds["personalBN_energy"] + ds["social_energy"] + ds["external_energy"]
ds["personalBN_nonfocal_energy"] = ds["personalBN_energy"] - ds["personalBN_focal_energy"] + ds["external_energy"]
for metric in all_metrics:
    # print("".join(["#"]*3)+f" {metric}")

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
    print(f"{metric2title[metric]} & ", end="")
    for r in ["compliant", "resilient", "resistant"]:
        print(f"${a.loc[a.response_type==r, metric].values[0]:.2f} \pm {a.loc[a.response_type==r, "sd"].values[0]:.2f}$", end=" & ")
    print("\\\\")
print(f"proportion & ", end="")
sum = np.sum([a.loc[a.response_type==r, "count"].values[0] for r in ["compliant", "resilient", "resistant"]])
for r in ["compliant", "resilient", "resistant"]:#a.response_type.unique():
    print(f"${(a.loc[a.response_type==r, "count"].values[0]/sum)*100:.1f}"+ "\\,\\%$", end=" & ")
print("\\\\")


# %%
# ----------------------------------------------
# -------    PLOT METRICS
# ----------------------------------------------
ds.sel(time=94.5).focal_belief.mean()
#%%

t=194.5
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
        axes[idx].set_ylabel(metric2title[metric])
    ttt=  f"{t-4.5}-{t+4.5}" if (t%10) == 4.5 else t
    name=metric+f" t={ttt}"
    fig.suptitle(name, y=1, va="top", fontsize=bigfs)
    plt.savefig( f"figs/characteristicsResponseTypes_{condition_string}_t{ttt}_{metric}.png")

# %%
# ----------------------------------------------
# -------    PLOT ENERGIES OVER TIME
# ----------------------------------------------
s_ext =4
energy_metrics = [ "energy", "personalBN_focal_energy", "personalBN_nonfocal_energy", "external_energy","social_energy",]
fig, axs = plt.subplots(2,3, sharex=True, sharey=True, figsize=(16/2.54, 8/2.54))
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
    a = a.loc[a.response_type.isin(["persistent-positive", "compliant", "resilient", "resistant"])]
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
    if metric == "social_energy":
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()

        axs[-1, -1].legend(
            handles,
            labels,
            loc="center",
            frameon=False,
            fontsize=smallfs
        )
    ax.set_title(metric2title[metric])
    ax.set_ylabel("energy")
    if metric=="external_energy":
        ax.text(125, ax.get_ylim()[0]+8, "external\npressure\n"+rf"$s={s_ext}$", va="center", ha="center", fontsize=smallfs)
axs[0, -1].tick_params(labelbottom=True)
axs[0, -1].set_xlabel("time")
axs[-1,-1].axis("off")

yl,yh = ax.get_ylim()
for ax in axs.flatten()[:-1]:
    ax.fill_between([100,150], [yl,yl], [yh,yh], color="red", alpha=(1+np.log2(s_ext))/5*0.4, zorder=-1, lw=0)
    ax.set_ylim(yl, yh)
fig.subplots_adjust(left=0.09, top=0.93, right=0.98, bottom=0.1)
plt.savefig( f"figs/energiesOverTime_{condition_string}_s{s_ext}.png")

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

    ax.set_ylim(-1.15,1.15)
fig.subplots_adjust(left=0.07, top=0.93, right=0.98, bottom=0.1)
plt.savefig( f"figs/focalBeliefsOverTime_{condition_string}.png")

# %%
