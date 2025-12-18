# %%
# ------------------------------------ #
# Analysis for Adaptive Networks model #
# 2025-12
# This file uses simulation output data from the adaptive network model and analyses:
# 1. the distribution of responses to external events
# 2. the characteristics of different response types
# The input data for one seed includes the belief_values and edge_weights of each agent for a number of time steps (or avg times)

#################################
#####  Plot resiliant/resistant/compliant frequency   #####
#################################


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

# %%

resultsfolder = "2025-12-16/"

T = 300
params = {
    "n_agents": 100,
    "belief_options": np.linspace(-1, 1, 21),
    "memory": 1,
    "M": 10,
    "starBN": False,
    "depolarisation": False,
    "focal_dim": "a",
    "link_prob": 10 / 100,
    "T": T,
    # "track_times": list(range(0, 10))
    # + list(range(90, 101))
    # + list(range(140, 151))
    # + list(range(190, 201))
    # + list(range(290, 301)),
    "external_event_times": list(range(100, 150)),
    "external_pressure": "a",
}

# Derived parameters
params["belief_dimensions"] = list(string.ascii_lowercase[: params["M"]])
params["edge_list"] = list(combinations(params["belief_dimensions"], 2))
params["edgeNames"] = [f"{i}_{j}" for i, j in params["edge_list"]]

pressures = dict(
    no_pressure=0,
    weak_focal=1,
    medium_focal=2,
    strong_focal=4,
    xxstrong_focal=8,
    xxxstrong_focal=16,
)


def generate_filename(params, results_folder):
    social_net = f"(p={params['link_prob']})"
    externalevent = f"_ext-{params['external_event_times'][0]}-{params['external_event_times'][-1]}-on-{params['external_pressure']}-strength{params['external_pressure_strength']}"
    return (
        f"{results_folder}adaptiveBN_M-{params['M']}{'star' if params['starBN'] else ''}-{'depolInitial' if params["depolarisation"] else 'randomInitialOps'}_n-{params['n_agents']}-{social_net}"
        f"_eps{params['eps']}-m{params['memory']}"
        f"_lam{params['lam']}_rho{params['rho']:.2f}_beta{params['beta']}_initialW-{params['initial_w']}"
        f"{externalevent}"
        f"_seed{params['seed']}"
    )


eps_adaptive = 1.0
lam_adaptive = 0.005
rho = 1.0 / 3.0
initial_w = 0.2
beta = 3.0

# %%

params["seed"] = 0
params["eps"] = eps_adaptive
params["lam"] = lam_adaptive
params["initial_w"] = initial_w
params["rho"] = rho
params["beta"] = beta
params["external_pressure_strength"] = pressures["no_pressure"]
filename = generate_filename(params, resultsfolder + "sims/")
df = pd.read_csv(filename + ".csv", low_memory=False)


#################################
#####  NEW   #####
#################################


# %%
seeds = list(range(100))
fixadaptive = [(0.0, 0.0), (eps_adaptive, lam_adaptive)]
Nruns = len(pressures) * len(seeds) * len(fixadaptive)

# Dimensions (run, time, agent_id, belief/belief1/belief2)
belief_results = {}  # will be (Nruns, n_times, n_agents, n_beliefs)
edge_results = {}  # will be (Nruns, n_times, n_agents, n_beliefs, n_beliefs)

# run metadata arrays
meta_seed = np.zeros(Nruns, dtype=int)
meta_adaptive = np.zeros(Nruns, dtype=bool)
meta_pressure_strength = np.zeros(Nruns, dtype=int)

# ===== main loop =====
run_idx = 0

params["initial_w"] = initial_w
params["rho"] = rho
params["beta"] = beta

for event, pressure_strength in pressures.items():
    params["external_pressure_strength"] = pressure_strength

    for seed in seeds:
        params["seed"] = seed

        for eps, lam in fixadaptive:
            # metadata
            meta_seed[run_idx] = seed
            meta_adaptive[run_idx] = eps > 0

            params["eps"] = eps
            params["lam"] = lam

            filename = generate_filename(params, resultsfolder + "sims/")
            df = pd.read_csv(filename + ".csv", low_memory=False)

            meta_pressure_strength[run_idx] = pressure_strength
            time_points = df.time.unique()
            for time_idx, t in enumerate(time_points):
                df_t = df.loc[df.time == t]

                # belief data
                for belief_idx, dim in enumerate(params["belief_dimensions"]):
                    if dim not in belief_results:
                        belief_results[dim] = np.zeros(
                            (Nruns, len(time_points), params["n_agents"]),
                            dtype=np.float32,
                        )
                    belief_results[dim][run_idx, time_idx, :] = (
                        df_t.set_index("agent_id")[dim]
                        .reindex(range(params["n_agents"]))
                        .values
                    )

                # edge data
                for edge in params["edgeNames"]:
                    if edge not in edge_results:
                        edge_results[edge] = np.zeros(
                            (Nruns, len(time_points), params["n_agents"]),
                            dtype=np.float32,
                        )
                    edge_results[edge][run_idx, time_idx, :] = (
                        df_t.set_index("agent_id")[edge]
                        .reindex(range(params["n_agents"]))
                        .values
                    )

                #######   TODO implement metrics here   ######

            run_idx += 1
            if run_idx % 10 == 0:
                print(run_idx, " of ", Nruns)

# ===== Build xarray Dataset =====

# Reshape attribute data: (run, time, agent_id, belief)
belief_array = np.stack(
    [belief_results[dim] for dim in params["belief_dimensions"]], axis=-1
)

# Reshape edge data: (run, time, agent_id, belief1, belief2)
# Assuming edge_labels are like "a_b", "a_c", "b_c", etc.
n_beliefs = len(params["belief_dimensions"])
n_edges = len(params["edgeNames"])
edge_array = np.zeros(
    (Nruns, len(time_points), params["n_agents"], n_edges),
    dtype=np.float32,
)

for n_edge, edge in enumerate(params["edgeNames"]):
    b1, b2 = edge.split("_")  # format "a_b"
    edge_array[:, :, :, n_edge] = edge_results[edge]

ds = xr.Dataset(
    {
        "belief_value": (("run", "time", "agent_id", "belief"), belief_array),
        "edge_weight": (("run", "time", "agent_id", "edge"), edge_array),
    },
    coords={
        "run": np.arange(Nruns),
        "time": time_points,
        "agent_id": range(params["n_agents"]),
        "belief": params["belief_dimensions"],
        "edge": params["edgeNames"],
        "seed": ("run", meta_seed),
        "adaptive": ("run", meta_adaptive),
        "s_ext": ("run", meta_pressure_strength),
    },
)

ds = ds.set_index(run=["adaptive", "seed", "s_ext"])
ds = ds.unstack("run")

ds.attrs.update(
    {
        "epsilon": eps_adaptive,
        "mu": 0,
        "lambda": lam_adaptive,
        "beta": params["beta"],
        "initial_w": initial_w,
        "memory": params["memory"],
        "M": params["M"],
        "starBN": params["starBN"],
        "depolarisationScenario": params["depolarisation"],
        "social_network_link_probability": params["link_prob"],
        "rho": params["rho"],
        "nr_belief_options": len(params["belief_options"]),
        "focal_dim": params["focal_dim"],
    }
)


# %%
#################################
#####  Get Compliant etc   #####
#################################
# Define time windows
focal_dim = params["focal_dim"]
t_beforeEvent = 94.5
t_inEvent = 144.5
t_postEvent_short = 194.5
t_postEvent_long = 294.5
eval_time = t_postEvent_short
beliefs_beforeEvent = ds.belief_value.sel(time=t_beforeEvent, belief=focal_dim)
beliefs_inEvent = ds.belief_value.sel(time=t_inEvent, belief=focal_dim)
beliefs_eval = ds.belief_value.sel(time=eval_time, belief=focal_dim)


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

response = xr.where(
    (beliefs_beforeEvent < 0) & (beliefs_inEvent < 0) & (beliefs_eval >= 0),
    response_map["late-compliant"],
    xr.where(
        (beliefs_beforeEvent < 0) & (beliefs_inEvent < 0) & (beliefs_eval < 0),
        response_map["resistant"],
        xr.where(
            (beliefs_beforeEvent < 0) & (beliefs_inEvent >= 0) & (beliefs_eval < 0),
            response_map["resilient"],
            xr.where(
                (beliefs_beforeEvent < 0)
                & (beliefs_inEvent >= 0)
                & (beliefs_eval >= 0),
                response_map["compliant"],
                xr.where(
                    (beliefs_beforeEvent >= 0) & (beliefs_eval >= 0),
                    response_map["persistent-positive"],
                    xr.where(
                        (beliefs_beforeEvent >= 0) & (beliefs_eval < 0),
                        response_map["non-persistent-positive"],
                        np.nan,
                    ),
                ),
            ),
        ),
    ),
)
ds["response_type"] = response.astype(np.uint8)
ds["response_type"].loc[dict(s_ext=0)] = response_map["NA"]

# %%
ds["s_ext"] = ds["s_ext"].astype(np.uint8)
ds["time"] = ds["time"].astype(np.float32)
ds["seed"] = ds["seed"].astype(np.uint8)
ds["agent_id"] = ds["agent_id"].astype(np.uint8)
ds["belief"]
ds.to_netcdf("processed_data/2025-12-16_modelAdaptiveBN_results.ncdf")
# %%
ds = xr.load_dataset("processed_data/2025-12-16_modelAdaptiveBN_results.ncdf")

#################################
#####  #################################
#####  #################################
#####  VISUALISE   #####
#################################   #####
#################################   #####
#################################

bigfs = 10
smallfs = 8
plt.rcParams.update({"font.size": bigfs})
plt.rcParams.update({"axes.titlesize": bigfs})
plt.rcParams.update({"axes.labelsize": bigfs})
plt.rcParams.update({"legend.fontsize": smallfs})
plt.rcParams.update({"xtick.labelsize": smallfs})
plt.rcParams.update({"ytick.labelsize": smallfs})

# %%
# ds = xr.open_dataset("2025-12-15_modelAdaptiveBN_results.ncdf")


cat_type = pd.CategoricalDtype(categories=responses, ordered=True)
svals = [1, 2, 8]

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
    [(0.6, 0.1), (0.2, 0.365), (1.2, 0.7), (0.2, 0.48), (0.2, 0.57), (0.0, 0.9)],
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
nonfocalbeliefs = [
    f"{bel}" for bel in params["belief_dimensions"] if not bel == focal_dim
]
focaledges = [f"{edge}" for edge in params["edgeNames"] if focal_dim in edge]
nonfocaledges = [f"{edge}" for edge in params["edgeNames"] if not focal_dim in edge]
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
