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

rho=1/3
link_prob = 0.1
condition_string =f"omega{omega0}_rho{rho:.2f}_beta{beta}_p{link_prob}"


ds = xr.load_dataset(f"processed_data/2025-12-30_modelAdaptiveBN_sensitivityAnalyses_processedData_metricsOnly/2025-12-30_modelAdaptiveBN_{condition_string}_results_metricsOnly.ncdf", engine="netcdf4")

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
# -------    RESPONSE TYPES: FREQUENCY
# ----------------------------------------------

T = 200
dim = "0"
adaptive = 1
ext_pressure_strength = 4
countResponses = []
omega0 = 0.2
rho=1/3
beta=1.5
link_prob = 0.1
params=[
    [omega0, rho, beta, link_prob] for omega0 in [0.2, 0.4] for rho in [1/3, 2/3] for beta in [1.5, 3., 6.] for link_prob in [0.05, 0.1, 0.2 ]
]
convert = lambda omega0, rho, beta, link_prob: fr"$\omega_0={omega0}$, $\rho={rho:.2f}$,"+"\n"+rf"$\beta={beta}$, $p={link_prob}$"
for omega0, rho, beta, link_prob in params:
    condition_string =f"omega{omega0}_rho{rho:.2f}_beta{beta}_p{link_prob}"
    ds = xr.load_dataset(f"processed_data/2025-12-30_modelAdaptiveBN_sensitivityAnalyses_processedData_metricsOnly/2025-12-30_modelAdaptiveBN_{condition_string}_results_metricsOnly.ncdf", engine="netcdf4")
    for adaptive in [0,1,2]:
        resp = ds.response_type.sel(adaptive=adaptive, s_ext=ext_pressure_strength)
        for response_type, r in response_map.items():
            count = (resp == r).sum().values
            countResponses.append(
                {
                    "condition_string": condition_string,
                    "adaptiveness": adaptive,
                    "response": response_type,
                    "count": int(count),
                }
            )

dff = pd.DataFrame(countResponses)
dff = dff.sort_values(["condition_string", "adaptiveness", "response"])
dff["normalized_count"] = dff["count"] / (len(ds.seed) * len(ds.agent_id))
dff["response"] = dff["response"].replace("NA", np.nan)
df = dff.dropna().copy()
ncols=3
nrows = 6
fig, axs = plt.subplots(nrows,ncols, figsize=(8 / 2.54, 20 / 2.54))
for n, (ax, (omega0, rho, beta, link_prob)) in enumerate(zip(axs.flatten(), params[0:])):
    condition_string =f"omega{omega0}_rho{rho:.2f}_beta{beta}_p{link_prob}"
    if n>0:
        axs.flatten()[n].sharex(axs.flatten()[n-1])
    subset = df.loc[df.condition_string == condition_string]
    grouped = (
        subset.groupby(["adaptiveness", "response"])["normalized_count"].sum().reset_index()
    )
    pivoted = grouped.pivot(
        index="adaptiveness",
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
    if n%ncols==0 and (n-n%ncols)/ncols==int(nrows/2):
        ax.set_ylabel(r"response type frequency (with $s=4$)", fontsize=bigfs, y=1)
    ax.set_title(convert(omega0, rho, beta, link_prob), fontsize=smallfs)
    ax.set_xticks(ax.get_xticks())
    ax.set_xlabel("")
    ax.set_ylim(-0.0, 1.0)
    ax.set_yticks([0,0.5,1])
    if not n%ncols==0:
        ax.set_yticklabels([])
        ax.set_ylabel("")
axs[-1, int(ncols/2)].set_xlabel(r"adaptiveness factor ($\epsilon=1.0$, $\lambda=0.005$)")
import string
# for n, ax in enumerate(axs.flatten()): 
#     ax.text(0.025, 0.975, string.ascii_uppercase[n], fontsize=12, fontdict={"weight":"bold"},va="top", ha="left", transform=ax.transAxes)
fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.17, wspace=0.1)
fig.tight_layout()
plt.savefig(
     f"figs/results_SensAna_omega0{0.2}_proportions_evaluatedAtT={eval_time}.png", dpi=300
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
