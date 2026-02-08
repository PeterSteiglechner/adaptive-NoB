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
dsExample = xr.load_dataset(f"processed_data/"
                     f"2026-01-21_modelAdaptiveBN_{condition_string}_results_metricsOnly_tresh{tresh}.ncdf", engine="netcdf4")

# %%
belief_dimensions = list(range(dsExample.attrs["M"]))
edgeNames = combinations(belief_dimensions, 2)
focal_dim = dsExample.attrs["focal_dim"]
eval_time = dsExample.attrs["evaluation_time_for_response"]
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
            ["#4DAF4A", "#A6D854", "#469FDB", "#B8DFF3", "#7A0177", "#E41A1C",  "#666666"],
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
        width = 0.15
        offsets = [-2.*width, -1.*width, -0*width, 1.*width, 2.*width] if len(betas)==5 else [-1.*width, -0*width, 1.*width]
        all_ds = []
        for beta, off in zip(betas, offsets):
            condition_string =f"omega{omega0}_rho{rho:.2f}_beta{beta}_p{p}"
            ds = xr.load_dataset(f"processed_data/"
                    f"2026-01-21_modelAdaptiveBN_{condition_string}_results_metricsOnly_tresh{tresh}.ncdf", engine="netcdf4")
            param_coords = {"beta":[ds.attrs['beta']], "initial_w":[ds.attrs["initial_w"]]}
            ds = ds.expand_dims(param_coords)
            all_ds.append(ds["response_type"])

        big_ds = xr.concat(all_ds, dim="beta")
        dff = big_ds.to_dataframe().reset_index().groupby(["seed", "s_ext", "adaptive", "beta"])["response_type"].value_counts().reset_index()
        dff["normalized_count"] = dff["count"] / (len(ds.agent_id))
        dff["response"] = dff.apply(lambda x: response_map_inv[x["response_type"]], axis=1)
        dff["response"] = dff["response"].replace("NA", np.nan)
        df = dff.dropna().copy()
        df["s_ext_log2"] = np.log2(df["s_ext"]).astype(int)
        print("seeds ",len(dff.seed))
        responsesNames = ['persistent-positive', 'non-persistent-positive', 'compliant', 'late-compliant', 'resilient', 'resistant']
        for beta, ofs in zip(betas, offsets):
            subset = df.loc[(df.adaptive == adaptive)&(df.beta==beta) ]
            sns.stripplot(subset, ax=ax, x="s_ext_log2", hue="response", y="normalized_count", jitter=True, palette=cmap, hue_order=responses, legend=False, size=1, alpha=0.2, dodge=True)
            avgs = subset.groupby(["response", "s_ext_log2"])["normalized_count"].median().reset_index()
            sns.stripplot(avgs, ax=ax, x="s_ext_log2", hue="response", y="normalized_count", jitter=True, palette=cmap, hue_order=responses, legend=False, size=3, alpha=0.8, dodge=True, marker="s")   
            for patch in ax.patches[-len(avgs)*len(tmp.columns):]:
                patch.set_x(patch.get_x() + off - width/2)
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
    #plt.savefig(f"figs/SA2_responseTypes_p{p}_tresh{tresh}.png", dpi=600)
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
for t in ds.time:
    sns.histplot(ds.sel(s_ext=1, adaptive=0,time=t, seed=0).to_dataframe()["focal_belief"], label=t.values)
plt.legend(title="t")
# %%
a=2
# %%
a
