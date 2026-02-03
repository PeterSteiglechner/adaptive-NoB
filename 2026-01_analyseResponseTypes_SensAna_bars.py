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
tresh=0.0
rho=1/3
link_prob = 0.1
omega0 = 0.1
beta=3.0
condition_string =f"omega{omega0}_rho{rho:.2f}_beta{beta}_p{link_prob}"
ds = xr.load_dataset(f"processed_data/"
                     f"2026-01-21_modelAdaptiveBN_{condition_string}_results_metricsOnly_tresh{tresh}.ncdf", engine="netcdf4")

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

colBefore = "#1B1B1B"
colAfter = "#6B6B6B"


# 5 figures: p
# 2 columns: fixed adaptive
# 2 rows: omega0
absmeanedgestot = []
pressures = [1, 2, 4]
betas = [1.5, 2.25, 3., 4.5, 6.]
for p in  [0.05, 0.1, 0.2, 0.5]:
    fig, axs = plt.subplots(3,2, sharex=True, sharey=True, figsize=(8/2.54, 7/2.54))
    params = [[omega0, [eps, lam]] for omega0 in [0.1,0.2,0.4] for eps,lam in [(0.0,0.0), (1.0,0.005)]]
    for n, (omega0, learning) in enumerate(params):
        countResponses = []
        eps, lam = learning
        adaptive = int(eps>0)
        ax = axs.flatten()[n]
        # s_ext 1 2 4 on x
        # beta 1.5,3,6 on x_sub
        for s_ext in pressures:
            for beta in betas:
                condition_string =f"omega{omega0}_rho{rho:.2f}_beta{beta}_p{p}"
                ds = xr.load_dataset(f"processed_data/"
                        #+2026-01-19_modelAdaptiveBN_sensitivityAnalyses_processedData_metricsOnly/
                        f"2026-01-21_modelAdaptiveBN_{condition_string}_results_metricsOnly_tresh{tresh}.ncdf", engine="netcdf4")
                resp = ds.response_type.sel(adaptive=adaptive, s_ext=s_ext)
                focalBeliefStd = ds.sel(adaptive=adaptive, s_ext=s_ext, time=94.5).focal_belief.std(dim="agent_id").mean(dim="seed")
                focalBeliefStdPost = ds.sel(adaptive=adaptive, s_ext=s_ext, time=194.5).focal_belief.std(dim="agent_id").mean(dim="seed")
                focalBeliefStdErr = ds.sel(adaptive=adaptive, s_ext=s_ext, time=94.5).focal_belief.std(dim="agent_id").std(dim="seed")
                focalBeliefStdPostErr = ds.sel(adaptive=adaptive, s_ext=s_ext, time=194.5).focal_belief.std(dim="agent_id").std(dim="seed")
                for response_type, r in response_map.items():
                    count = (resp == r).sum().values
                    countResponses.append(
                        {
                            "condition_string": condition_string,
                            "adaptiveness": adaptive,
                            "s_ext": s_ext,
                            "beta": beta,
                            "response": response_type,
                            "count": int(count),
                            "focalBeliefStd": focalBeliefStd.values,
                            "focalBeliefStdPost": focalBeliefStdPost.values,
                            "focalBeliefStdErr": focalBeliefStdErr.values,
                            "focalBeliefStdPostErr": focalBeliefStdPostErr.values,
                        }
                    )
                absmeanedgestot.append([adaptive, p, omega0, s_ext, beta, ds.sel(adaptive=adaptive, s_ext=s_ext, time=94.5)["bn_abs_meanedge_tot"].mean(dim="agent_id").mean(dim="seed").values])
        dff = pd.DataFrame(countResponses)
        dff = dff.sort_values(["adaptiveness","s_ext", "beta", "response"])
        dff["normalized_count"] = dff["count"] / (len(ds.seed) * len(ds.agent_id))
        dff["response"] = dff["response"].replace("NA", np.nan)
        df = dff.dropna().copy()
        grouped = (
            df.groupby(["adaptiveness", "s_ext", "beta",  "response"])["normalized_count"].sum().reset_index()
        )
        grouped["normalized_count"] = grouped["normalized_count"].fillna(0)


        width = 0.15
        offsets = [-2.*width, -1.*width, -0*width, 1.*width, 2.*width] if len(betas)==5 else [-1.*width, -0*width, 1.*width]

        responsesNames = ['persistent-positive', 'non-persistent-positive', 'compliant', 'late-compliant', 'resilient', 'resistant']
        for b, off in zip(betas, offsets):
            tmp = (
                grouped[grouped["beta"] == b]
                .pivot(index="s_ext", columns="response", values="normalized_count")
                .reindex(columns=responsesNames)
                .fillna(0)
            )

            tmp.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                width=width*0.8,
                position=0,   # important
                color=[cmap[s] for s in responsesNames],
                legend=False,
            )
            for ns, s in enumerate(df.s_ext.unique()):
                std = df.loc[(df.s_ext==s) & (df.beta==b), "focalBeliefStd"].values[0]
                stderr = df.loc[(df.s_ext==s) & (df.beta==b), "focalBeliefStdErr"].values[0]
                ax.errorbar(ns + off-0.02, std, 
                            yerr=stderr,marker="d", color=colBefore, lw=0, markersize=1)
                stdPost = df.loc[(df.s_ext==s) & (df.beta==b), "focalBeliefStdPost"].values[0]
                stdPosterr = df.loc[(df.s_ext==s) & (df.beta==b), "focalBeliefStdPostErr"].values[0]
                ax.errorbar(ns + off+0.02, stdPost, 
                            yerr=stdPosterr,marker="d", color=colAfter, lw=0, markersize=1)

            # shift bars manually
            for patch in ax.patches[-len(tmp)*len(tmp.columns):]:
                patch.set_x(patch.get_x() + off - width/2)
        if ax==axs[0,-1]:
            ax.text( ns+0.07,1.23, "focal std", fontsize=smallfs-1, va="bottom", ha="center")
            ax.annotate(r"before", (ns+off,std), (ns-0.25,1.1), fontsize=smallfs-1, va="bottom", ha="center", arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=-.3"), bbox=dict(pad=0.0,fc='none', ec='none', color=colBefore), color=colBefore)
            ax.annotate(r"after", (ns+off,stdPost), (ns+0.3,1.1), fontsize=smallfs-1, va="bottom", ha="center", arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=-.3", color=colAfter), bbox=dict(pad=0.0,fc='none', ec='none'), color=colAfter)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0,1)
        ax.set_yticks([])
        ax.set_xticklabels([f"$2^{np.log2(x):.0f}$" for x in pressures], rotation=0)
        ax.set_title(fr"{'adaptive' if eps>0 else 'fixed'}", fontsize=smallfs, x=0.75 if eps==0 else 0.25, ha="right" if eps==0 else "center",y=1.05, transform=ax.transAxes)
        if ax ==axs[0,0]:
            for beta, os in zip(betas, offsets):
                ax.annotate(fr"{beta}", (os, 1), (1.4*os, 1.2), ha="left", rotation=60, fontsize=smallfs-1, arrowprops=dict(arrowstyle="-"), bbox=dict(pad=0.0,fc='none', ec='none'))
            ax.text(2 *offsets[0], 1.25, r"$\mathbf{\beta}$", ha="center", rotation=0, fontsize=smallfs, weight="bold")

        ax.set_xlabel(r"external pressure $s$",fontsize=smallfs,va="center")
        if eps==0:
            ax.set_ylabel(fr"$\omega_0={omega0}$", fontsize=smallfs, va="center")
    fig.suptitle(rf"link probability $p={p}$", fontsize=smallfs, x=0.57)
    fig.subplots_adjust(left=0.08, right=0.98, hspace=0.45, bottom=0.11, wspace=0.1)
    plt.savefig(f"figs/SA2_responseTypes_p{p}_tresh{tresh}.png", dpi=600)
#%%

# ----------------------------------------------
# -------    BN EDGES OVER SA params
# ----------------------------------------------

a = pd.DataFrame(absmeanedgestot,columns=["adaptive", "p", "omega0", "s_ext", "beta", "Omega"]).drop(columns="s_ext").groupby(["adaptive", "beta", "omega0","p"])["Omega"].mean().reset_index()

fig, axs = plt.subplots(3,2, figsize=(12/2.54, 6/2.54), sharex=True, sharey=True)
n=0
for omega0 in [0.1,0.2,0.4]:
    for ad in [0,1]:
        ax=axs.flatten()[n]
        aa = a.loc[(a.omega0==omega0) & (a.adaptive==ad) ]
        sns.barplot(aa, x="beta", y="Omega", ax=ax, legend=False, hue="p", palette="plasma")
        if ad==0:
            ax.text(0.97, 0.97, rf"$\omega_0={omega0}$", ha="right", va="top", transform=ax.transAxes, fontsize=bigfs, )
        n=n+1
        ax.set_ylabel("")
for row in [1]:
    axs[row,0].set_ylabel(r"BN-$|\Omega|$")
axs[0,0].set_title("fixed", fontsize=bigfs, y=1.02, x=0.5)
axs[0,1].set_title("adaptive", fontsize=bigfs, y=1.02, x=0.5)
axs[-1,0].set_xlabel(r"$\beta$", fontsize=bigfs)
axs[-1,1].set_xlabel(r"$\beta$", fontsize=bigfs)

ax =axs[0,0]
ps = aa.p.unique()
width=0.8
width=width/len(ps)
-0.3 -0.1
for os, p in zip([k*width+width/2 for k in range(-2,2)], ps):
    ax.annotate(fr"{p}", (os, 0.15), (1.4*os, 0.4), ha="left", rotation=60, fontsize=smallfs-1, arrowprops=dict(arrowstyle="-"), bbox=dict(pad=0.0,fc='none', ec='none'))
ax.text(0.15, 0.8, r"$\mathbf{p}$", ha="center", rotation=0, fontsize=bigfs, weight="bold")
fig.subplots_adjust(bottom=0.17, right=0.99, top=0.9, left=0.09)
plt.savefig("figs/2026-01-21_SA_BNOmega_over_beta.png", dpi=600)
# %%

# ----------------------------------------------
# -------    TEST
# ----------------------------------------------

condition_string =f"omega{0.1}_rho{rho:.2f}_beta{1.5}_p{0.1}"
ds = xr.load_dataset(f"processed_data/"
                     f"2026-01-21_modelAdaptiveBN_{condition_string}_results_metricsOnly_tresh{tresh}.ncdf", engine="netcdf4")
# %%
plt.figure(figsize=(10/2.54, 5/2.54))
for t in ds.time:
    sns.histplot(ds.sel(s_ext=1, adaptive=0,time=t, seed=0).to_dataframe()["focal_belief"], label=t.values, bins=np.linspace(-1.05,1.05,21))
plt.legend(title="t")
# %%


# ----------------------------------------------
# -------    One at a a time
# ----------------------------------------------

# %%
sext = [4]
initial_ws = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0]
betas = [0.5, 1.5, 2.25, 3., 4.5, 6., 10.]
adaptivefixed = [(1.0,0.005)] # [(0.0,0.0), (1.0,0.005)]
ps = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

allparams = [
    ("p", [[4], [0.2], [3.], ps]),
    ("beta", [[4], [0.2], betas, [0.1]]),
    ("omega0", [[4], initial_ws, [3.], [0.1]])
]


fig, axs = plt.subplots(2,3, figsize=(12/2.54, 8/2.54), sharex="col", sharey=True)
for row, adaptivefixed in enumerate([(0.0,0.0), (1.0,0.005)]):
    countResponses = []
    for col, (name, allpar) in enumerate(allparams):
        currsext, currinitws, currbetas, currps = allpar 
        for p in  currps:
            for omega0 in currinitws:
                for s_ext in currsext:
                    for beta in currbetas:
                        condition_string =f"omega{omega0}_rho{rho:.2f}_beta{beta}_p{p}"
                        ds = xr.load_dataset(f"processed_data/"
                                #+2026-01-19_modelAdaptiveBN_sensitivityAnalyses_processedData_metricsOnly/
                                f"2026-01-21_modelAdaptiveBN_{condition_string}_results_metricsOnly_tresh{tresh}.ncdf", engine="netcdf4")
                        eps, lam = adaptivefixed
                        adaptive = int(eps>0)
                        resp = ds.response_type.sel(adaptive=adaptive, s_ext=s_ext)
                        focalBeliefStd = ds.sel(adaptive=adaptive, s_ext=s_ext, time=94.5).focal_belief.std(dim="agent_id").mean(dim="seed")
                        focalBeliefStdPost = ds.sel(adaptive=adaptive, s_ext=s_ext, time=194.5).focal_belief.std(dim="agent_id").mean(dim="seed")
                        focalBeliefStdErr = ds.sel(adaptive=adaptive, s_ext=s_ext, time=94.5).focal_belief.std(dim="agent_id").std(dim="seed")
                        focalBeliefStdPostErr = ds.sel(adaptive=adaptive, s_ext=s_ext, time=194.5).focal_belief.std(dim="agent_id").std(dim="seed")
                        for response_type, r in response_map.items():
                            count = (resp == r).sum().values
                            countResponses.append(
                                {
                                    "name":name,
                                    "condition_string": condition_string,
                                    "s_ext":s_ext,
                                    "beta":beta,
                                    "p":p,
                                    "omega0":omega0,
                                    "adaptive": adaptive,
                                    "response": response_type,
                                    "count": int(count),
                                    "focalBeliefStd": focalBeliefStd.values,
                                    "focalBeliefStdPost": focalBeliefStdPost.values,
                                    "focalBeliefStdErr": focalBeliefStdErr.values,
                                    "focalBeliefStdPostErr": focalBeliefStdPostErr.values,
                                }
                            )
        dff = pd.DataFrame(countResponses)
        
        for name, ax in zip(dff.name.unique(), axs[row,:]):
            currdff = dff.loc[dff.name==name]
            currdff = currdff.sort_values([name])
            currdff["normalized_count"] = currdff["count"] / (len(ds.seed) * len(ds.agent_id))
            currdff["response"] = currdff["response"].replace("NA", np.nan)
            df = currdff.dropna().copy()
            grouped = (
                df.groupby([name,  "response"])["normalized_count"].sum().reset_index()
            )
            grouped["normalized_count"] = grouped["normalized_count"].fillna(0)            
            tmp = (
                grouped
                .pivot(index=name, columns="response", values="normalized_count")
                .reindex(columns=responsesNames)
                .fillna(0)
            )
            tmp.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                color=[cmap[s] for s in responsesNames],
                legend=False,
            )
            stds = df.groupby(["beta", "p", "omega0"])["focalBeliefStd"].first().values
            stderrs = df.groupby(["beta", "p", "omega0"])["focalBeliefStdErr"].first().values
            stdPosts = df.groupby(["beta", "p", "omega0"])["focalBeliefStdPost"].first().values
            stdPosterrs = df.groupby(["beta", "p", "omega0"])["focalBeliefStdPostErr"].first().values
            ax2 = ax.twinx()
            ax2.set_ylim(0,2)
            ax2.set_yticks([0,1,2])
            for ns, (std, stderr, stdpost, stdposterr) in enumerate(zip(stds, stderrs, stdPosts, stdPosterrs)):
                ax2.errorbar(ns, std, 
                            yerr=stderr,marker="d", color=colBefore, lw=1, markersize=3)
                ax2.errorbar(ns +0.02, stdpost, 
                            yerr=stdposterr,marker="d", color=colAfter, lw=1, markersize=3)
            if ax==axs[0,-1]:
                ax.text( ns+0.07,1.18, "focal std", fontsize=smallfs-1, va="bottom", ha="center")
                ax2.annotate(r"before", (ns,std), (ns-1,2.17), fontsize=smallfs-1, va="bottom", ha="center", arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=-.3"), bbox=dict(pad=0.0,fc='none', ec='none', color=colBefore), color=colBefore)
                ax2.annotate(r"after", (ns,stdpost), (ns+0.7,2.17), fontsize=smallfs-1, va="bottom", ha="center", arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=-.2", color=colAfter), bbox=dict(pad=0.0,fc='none', ec='none'), color=colAfter)
            if ax==axs[0,-1]:
                ax2.set_ylabel("focal belief standard deviation (dots)", y=-0.1)
            else:
                ax2.set_ylabel("")
                ax2.set_yticklabels([])
            ax.set_ylim(0,1)
            label={
                "p": r"link probability $p$",
                "beta": "attention " + r"$\beta$",
                "omega0": r"initial edge weight $\omega_0$",
            }
            ax.set_yticks([0,0.5,1.])
            ax.set_xlabel(label[name], fontsize=bigfs, va="top",)
axs[0,0].set_ylabel("Proportion (bars)", fontsize=bigfs, y=0)
axs[0,1].set_title("fixed belief networks", fontsize=bigfs)
axs[1,1].set_title("adaptive belief networks", fontsize=bigfs)
fig.suptitle(rf"One-Factor-At-a-Time Sensitivity Analysis (for $s={s_ext}$)", fontsize=bigfs)
fig.subplots_adjust(hspace=0.35, top=0.85, left=0.1, right=0.9, bottom=0.18)
plt.savefig("figs/sensAna_OFAT_full.png", dpi=600)
        

# %%
"""
Take home:
* social connectivity p:
    - increasing p reduces consensus in fixed and adaptive belief networks
    - also increases social pressure and thereby helps to "lock-in" people after the external pressure event. 
    - however, when consensus is reached (around p=0.3), agents fully resist (if they hold a shared negative focal beliefs before the external event). 
* Attention $\beta$:
    - when agents do not pay attention to dissonance, they do not adhere to the external event much (resilient = compliant = non-persistent-positives = persistent positives). 
    - when agents pay a lot of attention to dissonance, resilience grows
    - holds for fixed and adaptive. but with adaptive belief networks, we always retain a few resilient agents
* initial edge weight (for adaptive) and fixed edge weight (for fixed), $\omega_0$:
    - for fixed: consensus breaks at critical omega0. 
    - for fixed: when consensus breaks, more resilience (and if omega0 even stronger, more resistance)
    - for adaptive, omega0 matters little, unless it is very strong. Then it fosters resilience and resistance.
"""
# %%
