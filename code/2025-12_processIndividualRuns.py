# %%
# ------------------------------------ #
# Analysis for Adaptive Networks model #
# 2025-12
# This file uses simulation output data from the adaptive network model and combines it into a single netcdf file. analyses:
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
import pandas as pd
import numpy as np
import json
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import xarray as xr

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
    "focal_dim": "0",
    "link_prob": 10 / 100,
    "T": T,
    # "track_times": list(range(0, 10))
    # + list(range(90, 101))
    # + list(range(140, 151))
    # + list(range(190, 201))
    # + list(range(290, 301)),
    "external_event_times": list(range(100, 150)),
    "external_pressure": "0",
}

# Derived parameters
params["belief_dimensions"] = [str(i) for i in range(params["M"])]
# list(string.ascii_lowercase[: params["M"]])
params["edge_list"] = list(combinations(params["belief_dimensions"], 2))
params["edgeNames"] = [f"{i}_{j}" for i, j in params["edge_list"]]

pressures = dict(
    no_pressure=0,
    # weak_focal=1,
    # medium_focal=2,
    # strong_focal=4,
    # xxstrong_focal=8,
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
seeds = list(range(20))
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
ds.attrs["evaluation_time_for_response"] = eval_time
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
# %%
