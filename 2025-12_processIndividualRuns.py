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
import time
import json
from scipy.spatial.distance import pdist, squareform
# from sklearn.manifold import MDS
import networkx as nx
import xarray as xr

# %%

resultsfolder = "sims/2026-01-16_singleRuns/"

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
params["edge_list"] = list(combinations(params["belief_dimensions"], 2))
params["edgeNames"] = [f"{i}_{j}" for i, j in params["edge_list"]]

pressures = dict(
    # no_pressure=0,
    weak_focal=1,
    medium_focal=2,
    strong_focal=4,
    # xxstrong_focal=8,
    # xxxstrong_focal=16,
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
condition_string="baselineConfig"
if condition_string != "baselineConfig":
    if  (not eps_adaptive == 1.0 and
        not lam_adaptive == 0.005 and 
        rho == 1.0 / 3.0 and
        initial_w == 0.2 and
        beta == 3.0):
        print("ERROR")
        condition_string+=time.time()


# %%
seeds = list(range(100))
fixadaptive = [(0.0, 0.0), (eps_adaptive, lam_adaptive)]
Nruns = len(pressures) * len(seeds) * len(fixadaptive)
n_agents = params["n_agents"]
n_beliefs = params["M"]
n_edges = len(params["edgeNames"])

# Dimensions (run, time, agent_id, belief/belief1/belief2)
belief_results = {}  # will be (Nruns, n_times, n_agents, n_beliefs)
edge_results = {}  # will be (Nruns, n_times, n_agents, n_beliefs, n_beliefs)

metrics = {} # will be (Nruns, n_times, n_agents, n_metrics)

meta_seed = np.zeros(Nruns, dtype=int)
meta_adaptive = np.zeros(Nruns, dtype=bool)
meta_pressure_strength = np.zeros(Nruns, dtype=int)

params["initial_w"] = initial_w
params["rho"] = rho
params["beta"] = beta
focal_dim = params["focal_dim"]
belief_dims = params["belief_dimensions"]
edge_names = params["edgeNames"]
triangles = list(combinations(belief_dims, 3))
triangles_focal = [t for t in combinations(belief_dims, 3) if params["focal_dim"] in t]
edge_lookup = {edge: idx for idx, edge in enumerate(edge_names)}
triangle_edge_idx = [
    (
        edge_lookup[f"{a}_{b}"],
        edge_lookup[f"{b}_{c}"],
        edge_lookup[f"{a}_{c}"],
    )
    for a, b, c in triangles
]

run_idx = 0

store_beliefAndEdges = False

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

            filename = generate_filename(params, resultsfolder )
            # filename += "_detailed"
            df = pd.read_csv(filename + ".csv", low_memory=False)
            meta_pressure_strength[run_idx] = pressure_strength
            time_points = df.time.unique()
            if run_idx==0:
                metrics["n_neighbours"] = np.zeros((Nruns, len(time_points), n_agents), dtype=np.uint16)
                metrics["personalBN_energy"] = np.zeros((Nruns, len(time_points), n_agents), dtype=np.float32)
                metrics["personalBN_focal_energy"] = np.zeros((Nruns, len(time_points), n_agents), dtype=np.float32)
                metrics["external_energy"] = np.zeros((Nruns, len(time_points), n_agents), dtype=np.float32)
                metrics["social_energy"] = np.zeros((Nruns, len(time_points), n_agents), dtype=np.float32)
                metrics["nr_balanced_triangles_tot"] = np.zeros((Nruns, len(time_points), n_agents), dtype=np.float32)
                metrics["nr_balanced_triangles_focal"] = np.zeros((Nruns, len(time_points), n_agents), dtype=np.float32)
                # metrics["bn_alpha"] = np.zeros((Nruns, len(time_points), n_agents), dtype=np.float32)
                metrics["bn_avg_weighted_clustering"] = np.zeros((Nruns, len(time_points), n_agents), dtype=np.float32)
                metrics["bn_betweenness_centrality"] = np.zeros((Nruns, len(time_points), n_agents), dtype=np.float32)
                metrics["bn_expected_influence"] = np.zeros((Nruns, len(time_points), n_agents), dtype=np.float32)
                metrics["bn_abs_meanedge_focal"] = np.zeros((Nruns, len(time_points), n_agents), dtype=np.float32)
                metrics["bn_abs_meanedge_tot"] = np.zeros((Nruns, len(time_points), n_agents), dtype=np.float32)
                metrics["abs_meanbelief_nonfocal"] = np.zeros((Nruns, len(time_points), n_agents), dtype=np.float32)
                metrics["focal_belief"] = np.zeros((Nruns, len(time_points), n_agents), dtype=np.float32)
            
            for time_idx, (time, df_t) in enumerate(df.groupby("time", sort=True)):
                dfi = df_t.sort_values("agent_id")
                
                # CURRENTLY DISABLED belief and edge data
                if store_beliefAndEdges:
                    for belief_idx, dim in enumerate(belief_dims):
                        if dim not in belief_results:
                            belief_results[dim] = np.zeros(
                                (Nruns, len(time_points), n_agents),
                                dtype=np.float32,
                            )
                        belief_results[dim][run_idx, time_idx, :] = (
                            dfi[dim].values
                        )

                    # edge data
                    for edge in edge_names:
                        if edge not in edge_results:
                            edge_results[edge] = np.zeros(
                                (Nruns, len(time_points), n_agents),
                                dtype=np.float32,
                            )
                        edge_results[edge][run_idx, time_idx, :] = (
                            dfi[edge].values
                        )

                # ----------------------------------------
                # --------    METRICS        -------------
                # ----------------------------------------
                metrics["n_neighbours"][run_idx, time_idx, :] = dfi["n_neighbours"]

                beliefs = dfi[belief_dims].values # (agents, n_beliefs)
                edges = dfi[edge_names].values  # (agents, n_edges)

                # ------ BELIEFS --------------
                metrics["abs_meanbelief_nonfocal"][run_idx, time_idx, :] = (
                    np.abs(
                        beliefs[:, [i for i, b in enumerate(belief_dims) if b != focal_dim]]
                    ).mean(axis=1)
                )

                metrics["focal_belief"][run_idx, time_idx, :] = beliefs[:, belief_dims.index(focal_dim)]                

                # ------ ENERGIES --------------
                # Personal Energy;   SLOW: metrics["personal_energy"][run_idx, time_idx, :] = df_t.set_index("agent_id").apply(lambda ag: - np.sum([ag[edge[0]] * ag[edge] * ag[edge[2]] for edge in params["edgeNames"]]) - 1* ag["external_pressure_strength"]*ag[focal_dim] , axis=1)
                edge_idx = [(belief_dims.index(a), belief_dims.index(b))
                            for a, b in params["edge_list"]]
                edge_energy = np.zeros(n_agents)
                for e, (i, j) in enumerate(edge_idx):
                    edge_energy += beliefs[:, i] * edges[:, e] * beliefs[:, j]
                metrics["personalBN_energy"][run_idx, time_idx, :] = -edge_energy
                # External Energy: 
                metrics["external_energy"][run_idx, time_idx, :] = - dfi["external_pressure_strength"].values * beliefs[:, belief_dims.index(focal_dim)]
                # Social Energy
                metrics["social_energy"][run_idx, time_idx, :] = dfi["socialEnergy"]
                # Personal Focal Energy; SLOW: metrics["focal_energy"][run_idx, time_idx, :] =  df_t.set_index("agent_id").apply(lambda ag: - np.sum([ag[edge[0]] * ag[edge] * ag[edge[2]] for edge in params["edgeNames"] if focal_dim in edge]) - 1* ag["external_pressure_strength"]*ag[focal_dim] , axis=1)
                edge_idx_focal = [(belief_dims.index(a), belief_dims.index(b))
                            for a, b in params["edge_list"] if a==focal_dim or b==focal_dim]
                focal_edge_indices = [idx for idx, (a, b) in enumerate(params["edge_list"]) 
                      if a == focal_dim or b == focal_dim]
                focal_edge_energy = np.zeros(n_agents)
                for local_idx, (i, j) in enumerate(edge_idx_focal):
                    global_edge_idx = focal_edge_indices[local_idx]
                    focal_edge_energy += beliefs[:, i] * edges[:, global_edge_idx] * beliefs[:, j]
                metrics["personalBN_focal_energy"][run_idx, time_idx, :] = (-focal_edge_energy)

                # ------ BALANCE -----------
                # SLOW: metrics["nr_balanced_triangles_tot"][run_idx, time_idx, :] = dfi.apply(lambda ag: np.sum([ag[a+"_"+b]*ag[b+"_"+c]*ag[a+"_"+c]>0 for a,b,c in triangles]), axis=1)
                tri_balance = np.zeros(n_agents)
                for a, b, c in triangles:
                    tri_balance += (
                        dfi[f"{a}_{b}"].values
                        * dfi[f"{b}_{c}"].values
                        * dfi[f"{a}_{c}"].values
                    ) > 0
                metrics["nr_balanced_triangles_tot"][run_idx, time_idx, :] = tri_balance
                # # SLOW: metrics["nr_balanced_triangles_focal"][run_idx, time_idx, :] = dfi.apply(lambda ag: np.sum([ag[a+"_"+b]*ag[b+"_"+c]*ag[a+"_"+c]>0 for a,b,c in triangles_focal]), axis=1)
                tri_focal_balance = np.zeros(n_agents)
                for a, b, c in triangles_focal:
                    tri_focal_balance += (
                        dfi[f"{a}_{b}"].values
                        * dfi[f"{b}_{c}"].values
                        * dfi[f"{a}_{c}"].values
                    ) > 0
                metrics["nr_balanced_triangles_focal"][run_idx, time_idx, :] = tri_focal_balance

                # ----- BN Structure ------------
                # |w_ij|
                edge_abs = np.abs(edges)
                metrics["bn_abs_meanedge_tot"][run_idx, time_idx, :] = edge_abs.mean(axis=1)
                metrics["bn_abs_meanedge_focal"][run_idx, time_idx, :] = (
                    np.abs(dfi[[e for e in edge_names if focal_dim in e]].values).mean(axis=1)
                )

                # alpha: triangle balance
                # prod = np.ones((n_agents, len(triangle_edge_idx)))
                # for k, (e1, e2, e3) in enumerate(triangle_edge_idx):
                #     prod[:, k] = edges[:, e1] * edges[:, e2] * edges[:, e3]
                # metrics["bn_alpha"][run_idx, time_idx, :] = np.sign(prod).mean(axis=1)
                
                # average weighted_clustering (defined via geometric mean of abs of edge weights)
                W = np.zeros((n_agents, n_beliefs, n_beliefs))
                for e, (i, j) in enumerate(params["edge_list"]):
                    ii = belief_dims.index(i)
                    jj = belief_dims.index(j)
                    W[:, ii, jj] = edges[:, e]
                    W[:, jj, ii] = edges[:, e]
                absW = np.abs(W)
                max_weight = absW.max(axis=(1, 2), keepdims=True) # shape preserved for normalising later
                absW_normalised = absW / max_weight
                clust = np.zeros((n_agents, n_beliefs))
                for i in range(n_beliefs):
                    for j in range(n_beliefs):
                        if j == i:
                            continue
                        for k in range(n_beliefs):
                            if k == i or k == j:
                                continue
                            clust[:, i] += (absW_normalised[:, i, j] * absW_normalised[:, i, k] * absW_normalised[:, j,k])**(1/3) 
                C = clust / ((n_beliefs-1) * (n_beliefs - 2))
                metrics["bn_avg_weighted_clustering"][run_idx, time_idx, :] = C.mean(axis=1)

                # expected influence on focal
                f = belief_dims.index(focal_dim)
                metrics["bn_expected_influence"][run_idx, time_idx, :] = (
                    W[:, f, :] * beliefs
                ).sum(axis=1)
                
                # betweenness centrality:  path length defined as 1/|edgeweight|
                bc = np.zeros(n_agents)
                eps = 1e-6
                for a in range(n_agents):
                    G = nx.Graph()
                    for i in range(n_beliefs):
                        for j in range(i + 1, n_beliefs):
                            w = absW[a, i, j]
                            if w > 0:
                                G.add_edge(i, j, weight=1.0 / (w + eps))
                    bc[a] = nx.betweenness_centrality(
                        G, weight="weight", normalized=True
                    )[belief_dims.index(focal_dim)]

                metrics["bn_betweenness_centrality"][run_idx, time_idx, :] = bc
            run_idx += 1
            if run_idx % 10 == 0:
                print(run_idx, " of ", Nruns)
# %%
# ===== Build xarray Dataset =====

# BELIEF AND EDGE DATA CURRENTLY DISABLED
if store_beliefAndEdges:
    # Reshape belief and edge data: (run, time, agent_id, belief1, belief2)
    belief_array = np.stack(
        [belief_results[dim] for dim in params["belief_dimensions"]], axis=-1
    )
    edge_array = np.zeros(
        (Nruns, len(time_points), n_agents, n_edges),
        dtype=np.float32,
    )
    for n_edge, edge in enumerate(params["edgeNames"]):
        edge_array[:, :, :, n_edge] = edge_results[edge]

metrics["n_neighbours"] = metrics["n_neighbours"].astype(np.uint16)
metrics["nr_balanced_triangles_tot"] = metrics["nr_balanced_triangles_tot"].astype(np.uint16)
metrics["nr_balanced_triangles_focal"] = metrics["nr_balanced_triangles_focal"].astype(np.uint16)
allData =  {
        "n_neighbours": (("run", "time", "agent_id"), metrics["n_neighbours"]),
        "personalBN_energy": (("run", "time", "agent_id"), metrics["personalBN_energy"]),
        "external_energy": (("run", "time", "agent_id"), metrics["external_energy"]),
        "social_energy": (("run", "time", "agent_id"), metrics["social_energy"]),
        "personalBN_focal_energy": (("run", "time", "agent_id"), metrics["personalBN_focal_energy"]),
        "nr_balanced_triangles_tot": (("run", "time", "agent_id"), metrics["nr_balanced_triangles_tot"]),
        "nr_balanced_triangles_focal": (("run", "time", "agent_id"), metrics["nr_balanced_triangles_focal"]),
        # "bn_alpha": (("run", "time", "agent_id"), metrics["bn_alpha"]),
        "bn_avg_weighted_clustering": (("run", "time", "agent_id"), metrics["bn_avg_weighted_clustering"]),
        "bn_betweenness_centrality": (("run", "time", "agent_id"), metrics["bn_betweenness_centrality"]),
        "bn_expected_influence": (("run", "time", "agent_id"), metrics["bn_expected_influence"]),
        "bn_abs_meanedge_focal": (("run", "time", "agent_id"), metrics["bn_abs_meanedge_focal"]),
        "bn_abs_meanedge_tot": (("run", "time", "agent_id"), metrics["bn_abs_meanedge_tot"]),
        "abs_meanbelief_nonfocal": (("run", "time", "agent_id"), metrics["abs_meanbelief_nonfocal"]),
        "focal_belief": (("run", "time", "agent_id"), metrics["focal_belief"]),
    }
coords = {
        "run": np.arange(Nruns),
        "time": time_points,
        "agent_id": range(n_agents),
        "seed": ("run", meta_seed),
        "adaptive": ("run", meta_adaptive),
        "s_ext": ("run", meta_pressure_strength),
    }
if store_beliefAndEdges:
    allData["belief_value"] =  (("run", "time", "agent_id", "belief"), belief_array )
    allData["edge_weight"] = (("run", "time", "agent_id", "edge"), edge_array) 
    coords["belief"]= params["belief_dimensions"]
    coords["edge"]= params["edgeNames"]
ds = xr.Dataset(
    allData,
    coords=coords,
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
        "M": n_beliefs,
        "starBN": int(params["starBN"]),
        "depolarisationScenario": int(params["depolarisation"]),
        "social_network_link_probability": params["link_prob"],
        "rho": params["rho"],
        "nr_belief_options": len(params["belief_options"]),
        "focal_dim": params["focal_dim"],
    }
)


# %%
#################################
#####  Get ResponseType etc   #####
#################################
# Define time windows
t_beforeEvent = 94.5 # 
t_inEvent = 144.5
t_postEvent_short = 194.5
t_postEvent_long = 294.5
eval_time = t_postEvent_short
if store_beliefAndEdges:
    beliefs_beforeEvent = ds.belief_value.sel(time=t_beforeEvent, belief=focal_dim)
    beliefs_inEvent = ds.belief_value.sel(time=t_inEvent, belief=focal_dim)
    beliefs_eval = ds.belief_value.sel(time=eval_time, belief=focal_dim)
else:
    beliefs_beforeEvent = ds.focal_belief.sel(time=t_beforeEvent)
    beliefs_inEvent = ds.focal_belief.sel(time=t_inEvent)
    beliefs_eval = ds.focal_belief.sel(time=eval_time)


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
fname =  f"processed_data/2025-12-29_modelAdaptiveBN_{condition_string}_results{'_detailed' if 'detailed' in filename else ''}_{'metricsOnly' if not store_beliefAndEdges else 'withBeliefsAndEdges'}.ncdf"
ds.to_netcdf(fname)
print(f"stored as {fname}")
# %%

# %%
ds.response_type
# %%
