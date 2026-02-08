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
import matplotlib as mpl
import matplotlib.patches as mpatches

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
tresh = 0.0
# ds = xr.load_dataset(f"processed_data/2025-12-29_modelAdaptiveBN_{condition_string}_results_metricsOnly.ncdf", engine="netcdf4")
ds = xr.load_dataset(f"processed_data/2026-01-21_modelAdaptiveBN_{condition_string}_results_metricsOnly_tresh{0.0}.ncdf", engine="netcdf4")
print("seeds: ",len(ds.seed))
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
responses.append("NA")
response_map_inv = {val: key for key, val in response_map.items()}
# cmap = dict(
#     zip(
#         responses+["NA"],
#         ["#7fc97f", "#fdc086", "#386cb0", "#beaed4", "#f0027f", "#bf5b17", "#666666"],
#     )
# )
cmap = dict(
    zip(
            responses+["NA"],
            ["#4CAF50", "#AED581", "#2196F3", "#90CAF9", "#9C27B0", "#F44336", "#9E9E9E"],
        )
    )

#%%
# ----------------------------------------------
# -------    NO PRESSURE: VARIANCE AND CONSENSUS
# ----------------------------------------------
adaptive=1
a = (ds.sel(adaptive=adaptive, s_ext=0, time=94.5)[
        "focal_belief"
    ]>=0).sum(dim="agent_id")
(a>=98).sum(dim="seed"), (a<=2).sum(dim="seed") 

ds.sel(adaptive=adaptive, s_ext=0, time=94.5)[
        "focal_belief"
    ].mean(dim="agent_id").mean(dim="seed")


#%%
# ----------------------------------------------
# -------    RESPONSE TYPES: FREQUENCY
# ----------------------------------------------

# examplesim = pd.read_csv("sims/adaptiveBN_M-10-randomInitialOps_n-100-(p=0.1)_eps1.0-m1_lam0.005_rho0.33_beta3.0_initialW-0.2_ext-100-149-on-0-strength4_seed98_detailed.csv")
ext_pressure_strength = 4
for examplesimadaptive in [False, True]:
    if examplesimadaptive:
        examplesim = pd.read_csv(f"sims/2026-01-21_singleRuns/detailed/adaptiveBN_M-10-randomInitialOps_n-100-(p=0.1)_eps1.0-m1_lam0.005_rho0.33_beta3.0_initialW-0.2_ext-100-149-on-0-strength{ext_pressure_strength}_seed98_detailed.csv")
    else:
        examplesim = pd.read_csv(f"sims/2026-01-21_singleRuns/detailed/adaptiveBN_M-10-randomInitialOps_n-100-(p=0.1)_eps0.0-m1_lam0.0_rho0.33_beta3.0_initialW-0.2_ext-100-149-on-0-strength{ext_pressure_strength}_seed98_detailed.csv")
    examplesim.loc[examplesim.time.isin([99,149,199])].pivot_table(values="0", index="time", columns="agent_id")
    T = 200
    dim = "0"
    seed = 0  # np.random.randint(10)
    window = 10
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

    # AX 2 and 3
    dff = ds.drop_sel(s_ext=0).response_type.to_dataframe()["response_type"].reset_index().groupby(["seed", "s_ext", "adaptive"])["response_type"].value_counts().reset_index()
    dff["normalized_count"] = dff["count"] / (len(ds.agent_id))
    dff["response"] = dff.apply(lambda x: response_map_inv[x["response_type"]], axis=1)
    dff["response"] = dff["response"].replace("NA", np.nan)
    df = dff.dropna().copy()
    df["s_ext_log2"] = np.log2(df["s_ext"]).astype(int)
    print("seeds ",len(dff.seed))

    fig, axs = plt.subplots(1,4, gridspec_kw={"width_ratios":[2,0.15, 1,1]}, figsize=(16 / 2.54, 6 / 2.54))
    axs[1].axis("off")
    axs[3].sharex(axs[2])
    for ax, adaptive in zip(axs[2:], [False, True]):
        subset = df.loc[df.adaptive == adaptive]
        sns.stripplot(subset, ax=ax, x="s_ext_log2", hue="response", y="normalized_count", jitter=True, palette=cmap, hue_order=responses, legend=False, size=1.2, alpha=0.2, dodge=True)
        avgs = subset.groupby(["response", "s_ext_log2"])["normalized_count"].median().reset_index()
        sns.stripplot(avgs, ax=ax, x="s_ext_log2", hue="response", y="normalized_count", jitter=True, palette=cmap, hue_order=responses, legend=False, size=4, alpha=0.8, dodge=True, marker="s")
        for i in [0.5,2.5]:
            ax.fill_between([i,i+1], [0,0],[1,1], color="gainsboro", zorder=-1)
        ax.set_ylabel("proportion in one simulation", fontsize=bigfs)
        ax.set_title("adaptive" if adaptive else "fixed", fontsize=bigfs)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            [f"$2^{int(float(v)):0d}$" for v in df.s_ext_log2.unique()], rotation=0
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
            edgecolor='white',
            alpha=0.8,
        )
    ap = dict(arrowstyle="-", connectionstyle="arc3,rad=0", color="black", shrinkA=0, shrinkB=0,)
    # axs[3].annotate("late-compliant", (0.3,0.68), (0.8,0.68), fontsize=smallfs, arrowprops=ap,bbox=bboxprops)
    # axs[3].annotate("non-persistent-\npositive", (0.3,0.4), (0.8,0.2), fontsize=smallfs, arrowprops=ap,bbox=bboxprops)
    if examplesimadaptive:
        axs[0].text(65, 0.18, "not shown: ",fontsize=smallfs-2, va="top", ha="right", bbox=dict(facecolor="white", alpha=0.6, pad=0.0))
        axs[0].text(66,0.18, "late-compliant",fontsize=smallfs-2, va="top", ha="left",bbox=bboxprops)
        bboxprops["facecolor"] = cmap["non-persistent-positive"]
        axs[0].text(36, 0.58, "not shown: ",fontsize=smallfs-2, va="top", ha="right", bbox=dict(facecolor="white", alpha=0.6, pad=0.0))
        axs[0].text(37, 0.58, "non-persistent-positive",fontsize=smallfs-2, va="top", ha="left",bbox=bboxprops)

    ax_main = axs[0]
    # for seed=98:
    resistant = 50 # resistant
    persistentpositive = 5
    resilient = 55
    compliant = 8
    # for seed = 8 
    # resistant = 21
    # persistentpositive = 67
    # resilient = 23
    # compliant = 93
    # latecompliant = np.nan
    # nonpersistentpos = np.nan
    ax_main.plot([],[],lw=0.7, alpha=0.4, label="single\nagent", color="grey")
    leg= ax_main.legend(loc="center left")
    df_pivot = df_pivot.loc[df_pivot.index <= t]
    df_pivot.plot(ax=ax_main, lw=0.5, alpha=0.4, legend=False, color="grey", label="_")
    # for i, name in zip([latecompliant, nonpersistentpos, resistant, resilient, compliant, persistentpositive],["late-compliant", "non-persistent-positive","resistant", "resilient", "compliant", "persistent-positive"]):
    latecompliant=np.nan
    nonpersistentpos= np.nan
    for i, name in zip([resistant, resilient, compliant, persistentpositive],["resistant", "resilient", "compliant", "persistent-positive"]):
        if examplesimadaptive or name=="compliant":
            df_pivot.T.loc[i].plot(ax=ax_main, lw=2 if i not in [latecompliant, nonpersistentpos] else 1, ls="-" if i not in [latecompliant, nonpersistentpos] else "--", color=cmap[name], alpha=0.8, legend=False, label="_", )
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
        [(20, 0.7), (110,-0.4),(60,0.3), (162, -0.1)],
        ["persistent-positive", "resistant", "compliant", "resilient"],
    ):
        bboxprops = dict(
            boxstyle="round",
            facecolor=cmap[type],
            edgecolor="white",
            alpha=0.8,
        )
        if examplesimadaptive or type=="compliant":
            axs[0].text(
                x,
                y,
                type,
                color="white",
                va="center",
                ha="left",
                bbox=bboxprops,
                fontsize=smallfs,
            )

    axs[0].text(130,0.5, "external\npressure", ha="center", fontsize=smallfs)
    axs[0].text(0.02,1.02,f"example simulation ({'adaptive' if examplesimadaptive else 'fixed'}, $s=$"+f"{ext_pressure_strength}{', smoothed' if window>0 else ''})", transform=ax_main.transAxes, ha="left", va="bottom", fontsize=smallfs)
    # fig.set_facecolor("pink")
    import string
    for n, ax in enumerate(axs[[0,2,3]]): 
        ax.text(0.025, 0.975, string.ascii_uppercase[n], fontsize=12, fontdict={"weight":"bold"},va="top", ha="left", transform=ax.transAxes)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.17, wspace=0.1)

    plt.savefig(
        f"figs/results_{condition_string}{'_fixed' if not examplesimadaptive else ''}_proportions_evaluatedAtT={eval_time}.png", dpi=300
    )

#%%
# ----------------------------------------------
# -------    OVER S_EXT EXAMPLE RUN
# ----------------------------------------------

storeall = {}
for examplesimadaptive in [False, True]:
    fig, axs = plt.subplots(2,3, sharex=True, sharey=True, figsize=(16/2.54, 9/2.54))
    storeall[examplesimadaptive] = []
    for ax, sext in zip(axs.flatten(), [0,1,2,4,8,16]):
        if examplesimadaptive:
            examplesim = pd.read_csv(f"sims/2026-01-21_singleRuns/detailed/adaptiveBN_M-10-randomInitialOps_n-100-(p=0.1)_eps1.0-m1_lam0.005_rho0.33_beta3.0_initialW-0.2_ext-100-149-on-0-strength{sext}_seed98_detailed.csv")
        else:
            examplesim = pd.read_csv(f"sims/2026-01-21_singleRuns/detailed/adaptiveBN_M-10-randomInitialOps_n-100-(p=0.1)_eps0.0-m1_lam0.0_rho0.33_beta3.0_initialW-0.2_ext-100-149-on-0-strength{sext}_seed98_detailed.csv")
        examplesim.loc[examplesim.time.isin([99,149,199])].pivot_table(values="0", index="time", columns="agent_id")
        
        T = 200
        dim = "0"
        adaptive = True
        seed = 0  # np.random.randint(10)
        window = 10
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
        storeall[examplesimadaptive].append(df_pivot)
        bboxprops = dict(
                boxstyle="round",
                facecolor=cmap["late-compliant"],
                edgecolor='white',
                alpha=0.8,
            )
        ap = dict(arrowstyle="-", connectionstyle="arc3,rad=0", color="black", shrinkA=0, shrinkB=0,)

        ax_main = ax
        # for seed=98:
        resistant = 50 # resistant
        persistentpositive = 5
        resilient = 55
        compliant = 8
        ax_main.plot([],[],lw=0.7, alpha=0.4, label="agent", color="grey")
        leg= ax_main.legend(loc="center left")
        df_pivot = df_pivot.loc[df_pivot.index <= t]
        df_pivot.plot(ax=ax_main, lw=0.5, alpha=0.4, legend=False, color="grey", label="_")
        for i, name in zip([resistant, resilient, compliant, persistentpositive],["resistant", "resilient", "compliant", "persistent-positive"]):
            if examplesimadaptive or name=="compliant":
                df_pivot.T.loc[i].plot(ax=ax_main, lw=2 if i not in [latecompliant, nonpersistentpos] else 1, ls="-" if i not in [latecompliant, nonpersistentpos] else "--", color=cmap[name], alpha=0.8, legend=False, label="_", )
        ax_main.set_xlabel("time", fontsize=bigfs)
        ax_main.set_ylabel("focal belief", fontsize=bigfs, va="top")
        ax_main.set_xlim(0, T)
        if len(dim) == 1:
            ax_main.set_ylim(-1, 1)
        ax_main.set_yticks([-1,0,1])
        ax_main.set_clip_on(False)
        leg = ax_main.get_legend()
        leg.set_bbox_to_anchor((0.05,0.2,0.3,0.3))
        if sext > 0 and t > 100:
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
                    alpha=int_colors[sext],
                    zorder=-1,
                    lw=0,
                )

            ax_main.text(130,0., rf"$s={sext}$", ha="center", fontsize=smallfs)

        if sext==4:
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
                if examplesimadaptive or type=="compliant": 
                    ax_main.text(
                        x,
                        y,
                        type,
                        color="white",
                        va="center",
                        ha="left",
                        bbox=bboxprops,
                        fontsize=smallfs,
                    )
    axs[-1,0].set_xlabel("")
    axs[-1,-1].set_xlabel("")
    axs[0,0].text(0.02,1.02,f"example simulation ({'adaptive' if examplesimadaptive else 'fixed'}{f', smoothed ({window} steps)' if window>0 else ''})", transform=axs[0,0].transAxes, ha="left", va="bottom", fontsize=smallfs)
    import string
    for n, ax in enumerate(axs.flatten()): 
        ax.text(0.025, 0.975, string.ascii_uppercase[n], fontsize=12, fontdict={"weight":"bold"},va="top", ha="left", transform=ax.transAxes)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.12, wspace=0.1)
    plt.savefig(f"figs/2026-01-26_responses_over_sext{'' if examplesimadaptive else '_fixed'}.png", dpi=600)

#%%
x = np.arange(145, 165)
ax= plt.axes()
ax.plot([],[],"-k", label="fixed")
ax.plot([],[],"--k", label="adaptive")
ax.legend()
pd.DataFrame({f"{s2}-{s1}": (pd.DataFrame(storeall[False][ns+1].iloc[x],) - pd.DataFrame(storeall[False][ns].iloc[x],)).abs().mean(axis=1) for ns, (s1,s2) in enumerate(zip([1,2,4,8,16], [0,1,2,4,8,16]))}).plot(ax=ax, colormap="viridis")
plt.legend(title="avg focal\nbelief difference \nbetween simulations\nwith pressures $s$:", ncol=2)
pd.DataFrame({f"{s2}-{s1}": (pd.DataFrame(storeall[True][ns+1].iloc[x],) - pd.DataFrame(storeall[True][ns].iloc[x],)).abs().mean(axis=1) for ns, (s1,s2) in enumerate(zip([1,2,4,8,16], [0,1,2,4,8,16]))}).plot(ls="--",ax=ax, legend=False, colormap="viridis")
plt.ylim(0.0,0.4)
#plt.yscale("log")
plt.ylabel(r"$|x_{foc, i}(t)_{s=s1} - x_{foc, i}(t)_{s=s2}|$, averaged over all agents")


#%%


# %%
# ----------------------------------------------
# -----    DESCRIBE METRICS STATS    ------
# ----------------------------------------------
from scipy.stats import ttest_ind
metric2title = dict(
    focal_belief = r"$x_{foc}$",
    abs_meanbelief_nonfocal = r"$|X_{non\text{-}foc}|$",
    n_neighbours = r"$|\mathcal{K}|$", 
    bn_abs_meanedge_tot = r"BN-$|\Omega|$",
    bn_abs_meanedge_focal = r"BN-$|\Omega_{foc}|$",
    nr_balanced_triangles_tot = r"BN-$\alpha$",
    nr_balanced_triangles_focal = r"BN-$\alpha_{foc}$",
    bn_avg_weighted_clustering = r"BN-clust",
    # bn_betweenness_centrality = r"BN-centr",
    # bn_expected_influence = r"$\langle\delta x_{foc}\rangle$", 
    personalBN_focal_energy = r"$D_{BN\text{-}foc}$",
    personalBN_nonfocal_energy = r"$D_{non\text{-}foc}$",
    social_energy = r"$D_{social}$",
    # external_energy = r"$D_{ext}$",    
    # energy = r"$D_{tot}$",
)
metric2titleVerb = dict(
    focal_belief = r"focal belief",
    abs_meanbelief_nonfocal = r"extremity non-focal",
    n_neighbours = r"nr social contacts", 
    bn_abs_meanedge_tot = r"BN connectedness",
    bn_abs_meanedge_focal = r"BN connectedness focal",
    nr_balanced_triangles_tot = r"BN balance",
    nr_balanced_triangles_focal = r"BN balance focal",
    bn_avg_weighted_clustering = r"BN clustering",
    # bn_betweenness_centrality = r"BN centrality",
    # bn_expected_influence = r"expected influence focal", 
    personalBN_focal_energy = r"focal BN dissonance",
    personalBN_nonfocal_energy = r"non-focal BN dissonance",
    social_energy = r"focal social dissonance",
    # external_energy = r"focal external dissonance",    
    # energy = r"total dissonance",
)
BNmetrics = ["nr_balanced_triangles_tot", "nr_balanced_triangles_focal", "bn_abs_meanedge_tot", "bn_abs_meanedge_focal","bn_avg_weighted_clustering"]

all_metrics = metric2title.keys()

t = 94.5
ttt=  f"{t-4.5}-{t+4.5}" if (t%10) == 4.5 else t
cohensd = []
for s_ext in [1,2,4,8,16]:
    print("".join(["#"]*50)+f"\n time = {ttt}\n"+"".join(["#"]*50)+f"\n s_ext = {s_ext}\n"+"".join(["#"]*50))
    print(" & "+" & ".join(["", "compliant", "resilient", "resistant"]+ [r"$d_{\text{Rr-C}}$", r"$d_{\text{R-r}}$"]) + " & \\\\ \\hline")
    ds["energy"] = ds["personalBN_energy"] + ds["social_energy"] + ds["external_energy"]
    ds["personalBN_nonfocal_energy"] = ds["personalBN_energy"] - ds["personalBN_focal_energy"] + ds["external_energy"]
    for metric in all_metrics:
        # print("".join(["#"]*3)+f" {metric}")
        metricvals = (
            ds
            .sel(adaptive=1, time=t, s_ext=s_ext)[[metric, "response_type"]]
            .to_dataframe()
            .reset_index()
        )
        a = (
            metricvals
            .groupby("response_type")[metric]
            .agg(mean="mean", sd="std", count="count")
            .rename(columns={"mean":metric})
            .reset_index()
        )
        a["response_type"] = a["response_type"].map(response_map_inv)
        #display(a)
        print(f"{metric2titleVerb[metric]} &  "+f"{metric2title[metric]} &  ", end="")
        for r in ["compliant", "resilient", "resistant"]:
            # def cohens_d_glass(a, b):
            #     return (a.mean() - b.mean()) / ((b.var())) ** 0.5
            if len(a.loc[a.response_type==r])>0:
                print(
                    f"${a.loc[a.response_type==r, metric].values[0]:.2f} \\pm {a.loc[a.response_type==r, "sd"].values[0]:.2f}$", 
                    end=" & "
                    )
            else:
                print(" ", end=" & ")
        if "ext" in metric:
            print(fr" & ", end=" & ")
        else:
            def cohens_d(a, b):
                return (a.mean() - b.mean()) / ((a.var() + b.var()) / 2) ** 0.5

            for g, baseg, name in zip([["resistant", "resilient"], ["resistant"]],["compliant", "resilient"], ["rR-C", "R-r"]):
                base = metricvals[metricvals["response_type"] == response_map[baseg]][metric]
                group = metricvals[metricvals["response_type"].isin([response_map[gg] for gg in g])][metric]
                if len(group)>0:
                    stat, p = ttest_ind(group, base, equal_var=False)
                    # r" $d_{"+fr"{name}" +r"}="+ 
                    d = cohens_d(group, base)
                    print(fr"${d:.1f}$ {fr'$^*$' if p<0.05 else ''}", end=" & ")
                else:
                    d=np.nan
                    print(fr"", end=" & ")
                    
                cohensd.append([s_ext, metric2title[metric], metric2titleVerb[metric], name, d, ])
        print("\\\\")
    print(f"proportion & & ", end="")
    for r in ["compliant", "resilient", "resistant"]:#a.response_type.unique():
        if r in a.response_type.values:
            sum = np.sum([0 if rr not in a["response_type"].values else a.loc[a.response_type==rr, "count"].values[0] for rr in ["compliant", "resilient", "resistant"]])
            print(f"${(a.loc[a.response_type==r, "count"].values[0]/sum)*100:.1f}"+ "\\,\\%$", end=" & ")
    print(" & \\\\")
    for _ in range(3):
        print("")

#%%
diss = True
coefs_rr_v = pd.read_csv(f"regression_results/coef_bin1_main_{'' if diss else 'no_'}diss.csv")
coefs_rr_v["type"] = "Rr-C"
coefs_r_r = pd.read_csv(f"regression_results/coef_bin2_main_{'' if diss else 'no_'}diss.csv")
coefs_r_r["type"] = "R-r"
coeffs = pd.concat([coefs_rr_v, coefs_r_r])
coeffs = coeffs.rename(columns={"pressureVal":"s"})
#%%
cmap = dict(zip([1,2,4,8,16], 
    [(0.0000, 0.4470, 0.7410), # | deep blue         |
    (0.8500, 0.3250 ,0.0980), # | orange            |
    (0.9290 ,0.6940 ,0.1250), 
    (0.4940, 0.1840, 0.5560),
    (0.4660, 0.6740, 0.1880),
    (0.3010, 0.7450, 0.9330),
    (0.6350, 0.0780, 0.1840),
]
))

cohensd_df = pd.DataFrame(cohensd, columns=["s", "metricMath", "metric", "type", "d"])
nmetric = len(cohensd_df.metric.unique())
fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(16/2.54, 8/2.54))
#axs[0].scatter([],[],marker="o", c="grey", s=5, edgecolor="grey", label="regression\ncoefficient")
dotpatch = mpl.lines.Line2D(
    [0], [0], marker='o', linestyle='None', markerfacecolor='lightgrey', markeredgecolor='grey', markeredgewidth=0.7, markersize=4, label="regression\ncoefficient")
patch = mpatches.Patch(color="grey", label=r"Cohen's $d$")
axs[0].legend(handles=[patch, dotpatch],fontsize=smallfs, borderpad=0.25, handlelength=1, handleheight=0.5, handletextpad=0.7, facecolor="gainsboro", loc="lower right", edgecolor="none", framealpha=0.5)
for n, ty in enumerate(["rR-C", "R-r"]):
    ax = axs[n]
    ss=[1,2,4] if ty=="R-r" else [1,2,4,8,16]
    barsdf = cohensd_df.loc[(cohensd_df.type==ty)&(cohensd_df.s.isin(ss))].sort_values("s", ascending=False)
    coeffs = coeffs.loc[~coeffs.term.str.contains("Intercept")]
    coeffs["metric"] = coeffs["term"].map(metric2titleVerb)
    points_df = (
        coeffs.groupby(["metric", "s"])["beta"].first().reset_index()
    )
    plot_df = barsdf.merge(points_df, on=["metric", "s"], how="left")
    metric_order = list(metric2titleVerb.values())[::-1]
    sns.barplot(plot_df, x="d", y="metric", hue="s", hue_order=ss[::-1], dodge=True, ax=ax, palette=cmap, legend=False, zorder=10, order=metric_order, 
    errorbar=None, alpha=0.8)
    ax2 = ax.twiny()
    sns.stripplot(
        data=plot_df.dropna(subset=["beta"]),
        x="beta", y="metric", hue="s",
        hue_order=ss[::-1], order=metric_order,
        dodge=True, jitter=False,
        palette=cmap,
        ax=ax2, legend=False,
        size=4, zorder=20,
        edgecolor="grey",
        linewidth=1
    )    
    ax2.set_xlabel("Regression Coefficient (dots)", fontsize=smallfs)
    if diss:
        ax2.set_xlim(-1,1)
    else:
        ax2.set_xlim(-0.7,0.7)
    ax.set_xlabel(r"Cohen's $d$ (bars)", fontsize=smallfs)
    ax.set_title("\nResistant/Resilient vs. Compliant" if ty=="rR-C" else "\nResistant vs. Resilient", fontsize=bigfs+1)
    ax.set_ylabel("")
    maxy = np.argsort([p.get_y() for p in ax.patches],)[-10:-5] if n==0 else np.argsort([p.get_y() for p in ax.patches],)[-6:-3]
    #ax.legend(title='pressure $s$', fontsize=smallfs, title_fontsize=smallfs, ncol=3, frameon=False)
    for kk, (barind, text) in enumerate(zip(maxy, [fr"$s={s}$" for s in ss[::-1]])):
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
            xytext=(1.2, y+(1.3 if n==0 else 1)-(0.55 if n==0 else 0.35)*(len(maxy)-kk)),
            ha="left",
            va="center",
            rotation=0,
            fontsize=smallfs-1,
            arrowprops=dict(
                arrowstyle="-",
                lw=1,
                connectionstyle="arc,rad=0.1",  # vertical then tilt
                color=bar_fc,
            ),
            bbox={"pad":0.02, "fc":"white", "ec":"none"},
            color=bar_fc,
            zorder=20,
        )

    ax.vlines([0], -0.5, nmetric+0.5, lw=0.5, color="grey")
    ax.hlines([0.5+k for k in range(nmetric)], -1.7, 1.7, lw=0.5, color="grey")
    ax.hlines([nmetric-2.5, nmetric-3.5,nmetric-8.5], -1.7, 1.7, lw=1.5, color="grey")
    ax.set_ylim(-0.5, nmetric-0.5)
    ax.set_xlim(-1.7, 1.7)
    ax2.set_ylim(ax.get_ylim()) 

# draw once so tick labels exist
fig.canvas.draw()

for label in axs[0].get_yticklabels():
    if label.get_text() in list(coeffs.metric.unique()):
        label.set_fontweight("bold")


fig.subplots_adjust(left=0.23, top=0.82, right=0.98, bottom=0.12)
plt.savefig(f"figs/cohensd_baseline{'_withDiss' if diss else ''}.png", dpi=600)



#%%
# def annotate_s_on_barplot(ax, barsinds, texts,yoffs, rotation=45, xoff=1):
#     """
#     Annotate each bar with its hue value (s=...) on a seaborn barplot.
#     Works when hue="s" and dodge=True.
#     """
#     for barind, text, yoff in zip(barsinds, texts, yoffs):
#         bar = ax.patches[barind]
#         h = bar.get_height()
#         if np.isnan(h):
#             continue

#         x = bar.get_x() + bar.get_width() / 2
#         y = h
#         bar_fc = bar.get_facecolor()
#         ax.annotate(
#             text,
#             xy=(x, y),
#             xytext=(x+xoff, yoff),
#             ha="left",
#             va="center",
#             rotation=rotation,
#             fontsize=smallfs-1,
#             arrowprops=dict(
#                 arrowstyle="-",
#                 lw=1,
#                 connectionstyle="angle3,angleA=0,angleB=-90",  # vertical then tilt
#                 color=bar_fc,
#             ),
#             bbox={"pad":0.02, "fc":"white", "ec":"none"},
#             color=bar_fc,
#             zorder=20
#         )



# cohensd_df = pd.DataFrame(cohensd, columns=["s", "metricMath", "metric", "type", "d"])
# nmetric = len(cohensd_df.metric.unique())
# fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(16/2.54, 7/2.54))
# for n, ty in enumerate(["rR-C", "R-r"]):
#     ss=[1,2,4] if ty=="R-r" else [1,2,4,8,16]
#     ax = axs[n]
#     sns.barplot(cohensd_df.loc[(cohensd_df.type==ty)&(cohensd_df.s.isin(ss))], y="d", x="metricMath", hue="s", dodge=True, ax=ax, palette=cmap, legend=False, zorder=10, order=list(metric2title.values()))
#     if n==0:
#         #ax.legend(title='pressure $s$', fontsize=smallfs, title_fontsize=smallfs, ncol=3, frameon=False)
#         annotate_s_on_barplot(ax, barsinds=[k*nmetric for k in range(len(ss))], texts=[fr"$s={s}$" for s in ss], yoffs=[-1.55, -1.35, -1.15,-0.95,-0.75], xoff=1.2, rotation=0)
#     ax.set_title("Resistant/Resilient vs. Compliant" if ty=="rR-C" else "Resistant vs. Resilient", fontsize=bigfs)
#     ax.hlines([0], -0.5, nmetric-0.5, lw=0.5, color="grey")
#     ax.vlines([0.5+k for k in range(nmetric)], -1.7, 1.7, lw=0.5, color="grey")
#     ax.vlines([1.5, 2.5, 8.5], -1.7, 1.7, lw=1.5, color="grey")
#     ax.set_xlim(-0.5, nmetric-0.5)
#     ax.set_ylim(-1.7, 1.7)
#     ax.set_xlabel("")
#     ax.set_ylabel(r"Cohen's $d$")
    
# axs[1].annotate(
#             r"eg. BN-$|\Omega_{foc}|_R \gg$"+"\n"+r"BN-$|\Omega_{foc}|_r$",
#             xy=(4,1.1),
#             xytext=(6,1.6),
#             ha="left",
#             va="top",
#             rotation=0,
#             fontsize=smallfs,
#             arrowprops=dict(
#                 arrowstyle="-",
#                 lw=1,
#                 connectionstyle="angle3,angleA=180,angleB=80",  # vertical then tilt
#             ),
#             bbox={"pad":0.02, "fc":"gainsboro", "ec":"none", "alpha":0.8},
#             zorder=20
#         ) 


# fig.autofmt_xdate(rotation=90, ha="center")
# fig.subplots_adjust(left=0.09, top=0.9, right=0.98, bottom=0.21, hspace=0.05, wspace=0.1)

# plt.savefig("figs/cohensd_baseline_vert.png", dpi=600)

#%%
times = [94.5, 194.5]
print("".join(["#"]*50)+f"\n CHANGE IN BN NETWORKS \n"+"".join(["#"]*50)+f"\n time = {times[0]} vs. {times[1]}\n"+"".join(["#"]*50)+f"\n s_ext = {s_ext}\n"+"".join(["#"]*50))
s_ext = 4
for metric in BNmetrics:
    aaa = []
    print(f"{metric2titleVerb[metric]} & {metric2title[metric]} & ", end="")
    for t in times:
        a = (
                ds
                .sel(adaptive=1, time=t, s_ext=s_ext)[[metric, "response_type"]]
                .to_dataframe()
                .reset_index()
                .groupby("response_type")[metric]
                .agg(mean="mean", sd="std", count="count")
                .rename(columns={"mean":metric})
                .reset_index()
            )
        a["response_type"] = a["response_type"].map(response_map_inv)
        aaa.append(a)
    
    for r in ["compliant", "resilient", "resistant"]:
        # for a in aaa:
        #     print(f"${a.loc[a.response_type==r, metric].values[0]:.2f} \\pm {a.loc[a.response_type==r, "sd"].values[0]:.2f}$", end=" & ")        
        print(f"${(aaa[1].loc[a.response_type==r, metric].values[0] / aaa[0].loc[a.response_type==r, metric].values[0]):.2f}$", end=" & ")        
    print("\\\\")
print(f"proportion & & ", end="")
for r in ["compliant", "resilient", "resistant"]:#a.response_type.unique():
    if r in a.response_type.values:
        sum = np.sum([0 if rr not in a["response_type"].values else a.loc[a.response_type==rr, "count"].values[0] for rr in ["compliant", "resilient", "resistant"]])
        print(f"${(a.loc[a.response_type==r, "count"].values[0]/sum)*100:.1f}"+ "\\,\\%$", end=" & ")
print(" & \\\\")

#%% 
s_ext = 4
energy_metrics = ['energy',
 'personalBN_focal_energy',
 'personalBN_nonfocal_energy',
 'external_energy',
 'social_energy']
t = 94.5
ttt=  f"{t-4.5}-{t+4.5}" if (t%10) == 4.5 else t
fig, axs = plt.subplots(2,1, sharex=True, figsize=(12/2.54,8/2.54 ))
for metric in ["energy"]:
    for ax, adaptive in zip(axs, [0,1]):
        a = (
                ds
                .sel(adaptive=adaptive, time=t, s_ext=s_ext)[[metric, "response_type"]]
                .to_dataframe()
                .reset_index()
            )
        a["response_type"] = a["response_type"].replace(response_map_inv)
        sns.histplot(a, x=metric, hue="response_type", palette=cmap, ax=ax, bins=np.arange(-32,6.1,2), kde=True, multiple="dodge", legend=False)
        ax.set_yticks([])
        ax.set_ylabel("")
axs[0].set_title("fixed belief networks", fontsize=bigfs)
axs[1].set_title("adaptive belief networks", fontsize=bigfs)
axs[1].set_xlabel(rf"Dissonance {metric2title[metric]} at" + rf" $t={ttt}$", fontsize=bigfs)
fig.tight_layout()
plt.savefig("figs/DtotDist_adaptiveFixed_s4.png", dpi=600)

# %%
# ----------------------------------------------
# -------    PLOT METRICS
# ----------------------------------------------
plot_characteristics = False
if plot_characteristics:
    responsesNames = ['persistent-positive',
    'non-persistent-positive',
    'compliant',
    'late-compliant',
    'resilient',
    'resistant']
    t=94.5
    cmap["NA"]="grey"
    for metric in all_metrics:
        print("".join(["#"]*10)+f".  {metric}")
        fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 7 / 2.54), sharey=True)
        for idx, adaptive_val in enumerate([0, 1]):
            data_subset = ds.sel(time=t, adaptive=adaptive_val, s_ext=[1,2,4,8,16])[["response_type", metric]].to_dataframe().reset_index()[["s_ext", "response_type", metric]]
            data_subset["response_type"] = data_subset["response_type"].map(response_map_inv)
            sns.boxplot(
                data=data_subset,
                x="s_ext",
                y=metric,
                hue="response_type",
                hue_order=responsesNames,
                ax=axes[idx],
                fliersize=0,
                palette=cmap,
                legend=adaptive_val == 0,
                boxprops={"edgecolor": "white", "lw":0 }
            )
            if adaptive_val == 0:
                h,l = axes[idx].get_legend_handles_labels()
                axes[idx].legend(h,l,title="",ncol=2,fontsize=7, loc="lower left", columnspacing=0.2, handletextpad=0.1, handlelength=1)
            sns.stripplot(
                data=data_subset,
                x="s_ext",
                y=metric,
                hue="response_type",
                hue_order=responsesNames,
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
further_plots = True
if further_plots:
    s_ext =4
    energy_metrics = [ "energy", "personalBN_focal_energy", "personalBN_nonfocal_energy", "external_energy","social_energy",]
    fig, axs = plt.subplots(2,3, sharex=False, sharey=True, figsize=(16/2.54, 8/2.54))
    for ax, metric in zip(axs.flatten(), energy_metrics):
        a = (
            ds
            .sel(adaptive=1, time=[4.5,94.5,144.5,194.5], s_ext=s_ext)[[metric, "response_type"]]
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
        if ax in axs[0,:]:
            ax.set_xticklabels([])
            ax.set_xlabel("")
    axs[0, -1].tick_params(labelbottom=True)
    axs[0, -1].set_xlabel("time")
    
    axs[-1,-1].axis("off")
    yl,yh = ax.get_ylim()
    for ax in axs.flatten()[:-1]:
        ax.fill_between([100,150], [yl,yl], [yh,yh], color="red", alpha=(1+np.log2(s_ext))/5*0.4, zorder=-1, lw=0)
        ax.set_ylim(yl, yh)
    fig.subplots_adjust(left=0.09, top=0.93, right=0.98, bottom=0.13, hspace=0.25)
    plt.savefig( f"figs/energiesOverTime_{condition_string}_s{s_ext}.png", dpi=600)

# %%
# %%
# ----------------------------------------------
# -------    PLOT FOCAL BELIEFS OVER TIME
# ----------------------------------------------
further_plots = True
if further_plots:
    metric = "focal_belief"
    times = [4.5,94.5,144.5,194.5]
    for currseeds, ts in zip([ds.seed, range(100)], [times, times+[294.5]]):
        dss = ds if ts[-1] < 200 else xr.load_dataset(f"processed_data/2026-01-21_modelAdaptiveBN_{condition_string}_results_metricsOnly_tresh{0.0}_seed0-100_T300.ncdf", engine="netcdf4")

        fig, axs = plt.subplots(2,3, sharex=True, sharey=True, figsize=(16/2.54, 9/2.54))
        for ax, s in zip(axs.flatten(), [0,1,2,4,8,16]):
            a = (
                    dss
                    .sel(adaptive=1, seed=currseeds, s_ext=s)[[metric, "response_type"]]
                    .sel(time=ts)
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
        fig.subplots_adjust(left=0.09, top=0.93, right=0.98, bottom=0.13)
        plt.savefig( f"figs/focalBeliefsOverTime_{condition_string}{'_T300' if ts[-1]>200 else ''}.png", dpi=600)

# %%
# ----------------------------------------------
# -------    VARIANCE OF RESPONSE
# ----------------------------------------------

further_plots = False
if further_plots: 
    a = ds.sel(s_ext=4, adaptive=1)["response_type"].to_dataframe()
    a["response_type"] = a["response_type"].map(response_map_inv)
    pd.DataFrame(a.groupby("seed")["response_type"].value_counts()).reset_index().pivot_table(columns="seed", index="response_type", values="count").mean(axis=1)

    fig, axs = plt.subplots(1,2, figsize=(12/2.54, 6/2.54), sharex=True, sharey=True)
    for ax, ad in zip(axs, [0, 1]):
        a = ds.sel(s_ext=4, adaptive=ad)["response_type"].to_dataframe()
        a["response_type"] = a["response_type"].map(response_map_inv)
        a["response_type"] = pd.Categorical(a["response_type"], categories = [r for r in response_map.keys() if r!="NA"])
        
        sns.histplot(pd.DataFrame(a.groupby("seed")["response_type"].value_counts()).reset_index().pivot_table(columns="seed", index="response_type", values="count").T/len(ds.agent_id), bins=np.arange(0,1.01, 0.1), alpha=1, palette=cmap, color="response_type", multiple="dodge", ax=ax, legend=ad, edgecolors='none')
        ax.set_title("fixed" if not ad else "adaptive", x=0.75 if ad else 0.25)
        ax.set_xlabel("frequency")
        leg = ax.get_legend()
        if leg:
            leg.set_title("")
    axs[0].set_ylabel("Nr of simulations")
    fig.suptitle(fr"$s=4$, baseline configuration", fontsize=bigfs)
    fig.subplots_adjust(bottom=0.17, right=0.98)
    plt.savefig("figs/Distribution_responsetypes_variance.png", dpi=600)

# %%
