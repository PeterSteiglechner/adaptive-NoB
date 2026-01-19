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
omega0 = 0.1
beta=3.0
condition_string =f"omega{omega0}_rho{rho:.2f}_beta{beta}_p{link_prob}"

ds = xr.load_dataset(f"processed_data/"
                     #+2026-01-19_modelAdaptiveBN_sensitivityAnalyses_processedData_metricsOnly/
                     f"2026-01-19_modelAdaptiveBN_{condition_string}_results_metricsOnly_SensAna2.ncdf", engine="netcdf4")

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


# 5 figures: p
# 2 columns: fixed adaptive
# 2 rows: omega0
pressures = [1, 2, 4]
betas = [1.5, 2.25, 3., 4.5, 6.]
for p in [0.05, 0.1, 0.2, 0.5]:
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
                        f"2026-01-19_modelAdaptiveBN_{condition_string}_results_metricsOnly_SensAna2.ncdf", engine="netcdf4")
                resp = ds.response_type.sel(adaptive=adaptive, s_ext=s_ext)
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
                        }
                    )
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
            # shift bars manually
            for patch in ax.patches[-len(tmp)*len(tmp.columns):]:
                patch.set_x(patch.get_x() + off - width/2)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0,1)
        ax.set_yticks([])
        ax.set_xticklabels([f"$2^{np.log2(x):.0f}$" for x in pressures], rotation=0)
        ax.set_title(fr"{'adaptive' if eps>0 else 'fixed'}", fontsize=smallfs, x=0.75, ha="right",y=1.05, transform=ax.transAxes)
        if ax ==axs[0,0]:
            for beta, os in zip(betas, offsets):
                ax.annotate(fr"{beta}", (os, 1), (1.4*os, 1.2), ha="left", rotation=60, fontsize=smallfs-1, arrowprops=dict(arrowstyle="-"), bbox=dict(pad=0.0,fc='none', ec='none'))
            ax.text(2 *offsets[0], 1.25, r"$\mathbf{\beta}$", ha="center", rotation=0, fontsize=smallfs, weight="bold")

        ax.set_xlabel("external pressure",fontsize=smallfs,va="center")
        if eps==0:
            ax.set_ylabel(fr"$\omega_0={omega0}$", fontsize=smallfs, va="center")
    fig.suptitle(rf"link probability $p={p}$", fontsize=smallfs, x=0.57)
    fig.subplots_adjust(left=0.08, right=0.98, hspace=0.45, bottom=0.11, wspace=0.1)
    plt.savefig(f"figs/SA2_responseTypes_p{p}.png", dpi=600)
#%%

