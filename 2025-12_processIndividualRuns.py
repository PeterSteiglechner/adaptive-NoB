#%%
import numpy as np
import pandas as pd
from itertools import combinations
import time
import networkx as nx
import xarray as xr
from multiprocessing import Pool, cpu_count
from functools import partial

# %%
SA = False
resultsfolder = "sims/2026-01-21_singleRuns/" if not SA else "sims/2026-01-16_singleRuns_SensAna2/"
tresh=0.0
T = 200

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
    "external_event_times": list(range(100, 150)),
    "external_pressure": "0",
}

# Derived parameters
params["belief_dimensions"] = [str(i) for i in range(params["M"])]
params["edge_list"] = list(combinations(params["belief_dimensions"], 2))
params["edgeNames"] = [f"{i}_{j}" for i, j in params["edge_list"]]


if SA:
    pressures = dict(
        # no_pressure=0,
        weak_focal=1,
        medium_focal=2,
        strong_focal=4,
        # xxstrong_focal=8,
        # xxxstrong_focal=16,
    )
else:
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


def process_single_run(args):
    """Process a single simulation run - this function will be parallelized"""
    (pressure_strength, seed, eps, lam, params, resultsfolder, 
     store_beliefAndEdges, focal_dim, belief_dims, edge_names, triangles, 
     triangles_focal) = args
        
    # Reconstruct params
    params = params.copy()
    params["external_pressure_strength"] = pressure_strength
    params["seed"] = seed
    params["eps"] = eps
    params["lam"] = lam
    
    edge_idx = [(belief_dims.index(a), belief_dims.index(b)) for a, b in params["edge_list"]]
    edge_idx_focal = [
            (belief_dims.index(a), belief_dims.index(b))
            for a, b in params["edge_list"] if a == focal_dim or b == focal_dim
        ]
    focal_edge_indices = [
            idx for idx, (a, b) in enumerate(params["edge_list"])
            if a == focal_dim or b == focal_dim
        ]


    n_agents = params["n_agents"]
    n_beliefs = params["M"]
    n_edges = len(edge_names)
    
    filename = generate_filename(params, resultsfolder)
    
    try:
        df = pd.read_csv(filename + ".csv", low_memory=False)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None
    
    time_points = df.time.unique()
    time_points = [0, 4.5, 94.5, 144.5,194.5]
    n_times = len(time_points)
    df = df.loc[df.time.isin(time_points)]
    
    # Initialize result arrays for this run
    result = {
        'meta': {'seed': seed, 'adaptive': int(eps), 'pressure_strength': pressure_strength},
        'time_points': time_points,
        'metrics': {
            'n_neighbours': np.zeros((n_times, n_agents), dtype=np.uint16),
            'personalBN_energy': np.zeros((n_times, n_agents), dtype=np.float32),
            'personalBN_focal_energy': np.zeros((n_times, n_agents), dtype=np.float32),
            'external_energy': np.zeros((n_times, n_agents), dtype=np.float32),
            'social_energy': np.zeros((n_times, n_agents), dtype=np.float32),
            'nr_balanced_triangles_tot': np.zeros((n_times, n_agents), dtype=np.uint16),
            'nr_balanced_triangles_focal': np.zeros((n_times, n_agents), dtype=np.uint16),
            'bn_avg_weighted_clustering': np.zeros((n_times, n_agents), dtype=np.float32),
            'bn_betweenness_centrality': np.zeros((n_times, n_agents), dtype=np.float32),
            'bn_expected_influence': np.zeros((n_times, n_agents), dtype=np.float32),
            'bn_abs_meanedge_focal': np.zeros((n_times, n_agents), dtype=np.float32),
            'bn_abs_meanedge_tot': np.zeros((n_times, n_agents), dtype=np.float32),
            'abs_meanbelief_nonfocal': np.zeros((n_times, n_agents), dtype=np.float32),
            'focal_belief': np.zeros((n_times, n_agents), dtype=np.float32),
        }
    }
    
    if store_beliefAndEdges:
        result['beliefs'] = np.zeros((n_times, n_agents, n_beliefs), dtype=np.float32)
        result['edges'] = np.zeros((n_times, n_agents, n_edges), dtype=np.float32)
    
    # Process each timepoint
    for time_idx, (time, df_t) in enumerate(df.groupby("time", sort=True)):
        dfi = df_t.sort_values("agent_id")
        
        beliefs = dfi[belief_dims].values
        edges = dfi[edge_names].values
        
        if store_beliefAndEdges:
            result['beliefs'][time_idx] = beliefs
            result['edges'][time_idx] = edges
        
        # Compute metrics (same as original)
        result['metrics']['n_neighbours'][time_idx] = dfi['n_neighbours']
        result['metrics']['abs_meanbelief_nonfocal'][time_idx] = np.abs(
            beliefs[:, [i for i, b in enumerate(belief_dims) if b != focal_dim]]
        ).mean(axis=1)
        result['metrics']['focal_belief'][time_idx] = beliefs[:, belief_dims.index(focal_dim)]
        
        # Personal Energy
        edge_energy = np.zeros(n_agents)
        for e, (i, j) in enumerate(edge_idx):
            edge_energy += beliefs[:, i] * edges[:, e] * beliefs[:, j]
        result['metrics']['personalBN_energy'][time_idx] = -edge_energy
        
        # External & Social Energy
        result['metrics']['external_energy'][time_idx] = (
            -dfi["external_pressure_strength"].values * beliefs[:, belief_dims.index(focal_dim)]
        )
        result['metrics']['social_energy'][time_idx] = dfi['socialEnergy']
        
        # Focal Energy
        focal_edge_energy = np.zeros(n_agents)
        for local_idx, (i, j) in enumerate(edge_idx_focal):
            global_edge_idx = focal_edge_indices[local_idx]
            focal_edge_energy += beliefs[:, i] * edges[:, global_edge_idx] * beliefs[:, j]
        result['metrics']['personalBN_focal_energy'][time_idx] = -focal_edge_energy
        
        # Balance
        tri_balance = np.zeros(n_agents)
        for a, b, c in triangles:
            tri_balance += (dfi[f"{a}_{b}"].values * dfi[f"{b}_{c}"].values * dfi[f"{a}_{c}"].values) > 0
        result['metrics']['nr_balanced_triangles_tot'][time_idx] = tri_balance
        
        tri_focal_balance = np.zeros(n_agents)
        for a, b, c in triangles_focal:
            tri_focal_balance += (dfi[f"{a}_{b}"].values * dfi[f"{b}_{c}"].values * dfi[f"{a}_{c}"].values) > 0
        result['metrics']['nr_balanced_triangles_focal'][time_idx] = tri_focal_balance
        
        # BN Structure
        edge_abs = np.abs(edges)
        result['metrics']['bn_abs_meanedge_tot'][time_idx] = edge_abs.mean(axis=1)
        result['metrics']['bn_abs_meanedge_focal'][time_idx] = np.abs(
            dfi[[e for e in edge_names if focal_dim in e]].values
        ).mean(axis=1)
        
        # Weighted Clustering
        W = np.zeros((n_agents, n_beliefs, n_beliefs))
        for e, (i, j) in enumerate(params["edge_list"]):
            ii = belief_dims.index(i)
            jj = belief_dims.index(j)
            W[:, ii, jj] = edges[:, e]
            W[:, jj, ii] = edges[:, e]
        absW = np.abs(W)
        max_weight = absW.max(axis=(1, 2), keepdims=True)
        absW_normalised = absW / max_weight
        clust = np.zeros((n_agents, n_beliefs))
        for i in range(n_beliefs):
            for j in range(n_beliefs):
                if j == i:
                    continue
                for k in range(n_beliefs):
                    if k == i or k == j:
                        continue
                    clust[:, i] += (
                        absW_normalised[:, i, j] * absW_normalised[:, i, k] * absW_normalised[:, j, k]
                    ) ** (1/3)
        C = clust / ((n_beliefs - 1) * (n_beliefs - 2))
        result['metrics']['bn_avg_weighted_clustering'][time_idx] = C.mean(axis=1)
        
        # Expected Influence
        f = belief_dims.index(focal_dim)
        result['metrics']['bn_expected_influence'][time_idx] = (W[:, f, :] * beliefs).sum(axis=1)
        
        # Betweenness Centrality
        bc = np.zeros(n_agents)
        eps_val = 1e-6
        for a in range(n_agents):
            G = nx.Graph()
            for i in range(n_beliefs):
                for j in range(i + 1, n_beliefs):
                    w = absW[a, i, j]
                    if w > 0:
                        G.add_edge(i, j, weight=1.0 / (w + eps_val))
            bc[a] = nx.betweenness_centrality(
                G, weight="weight", normalized=True
            )[belief_dims.index(focal_dim)]
        result['metrics']['bn_betweenness_centrality'][time_idx] = bc
    
    return result
#%%
if __name__ == '__main__':
    seeds = list(range(100 if SA else 300))
    fixadaptive = [(0.0, 0.0), (1.0, 0.005)]
    store_beliefAndEdges = False

    if SA:
        param_combis = [
            [omega0, 1.0 / 3.0, beta, link_prob]
            for omega0 in [0.1, 0.2, 0.4]
            for beta in [1.5, 2.25, 3.0, 4.5, 6.0]
            for link_prob in [0.05, 0.1, 0.2, 0.5, 1]
        ]
        pressures = {n:p for n,p in pressures.items() if p in [1,2,4]}
    else:
        param_combis = [
            [omega0, 1.0 / 3.0, beta, link_prob]
            for omega0 in [0.2]
            for beta in [3.0]
            for link_prob in [0.1]
        ]

    focal_dim = params["focal_dim"]
    belief_dims = params["belief_dimensions"]
    edge_names = params["edgeNames"]
    triangles = list(combinations(belief_dims, 3))
    triangles_focal = [t for t in combinations(belief_dims, 3) if params["focal_dim"] in t]

    for initial_w, rho, beta, link_prob in param_combis:
        params["initial_w"] = initial_w
        params["rho"] = rho
        params["beta"] = beta
        params["link_prob"] = link_prob
        condition_string = f"omega{initial_w}_rho{rho:.2f}_beta{beta}_p{link_prob}" if SA else "baselineConfig"
        print(f"\nProcessing: {condition_string}")
        
        # Build arguments for all runs
        args_list = []
        for event, pressure_strength in pressures.items():
            for seed in seeds:
                for eps, lam in fixadaptive:
                    args_list.append((
                        pressure_strength, seed, eps, lam, params, resultsfolder,
                        store_beliefAndEdges, focal_dim, belief_dims, edge_names,
                        triangles, triangles_focal
                    ))
        
        # Parallel processing
        n_processes = cpu_count() - 3  # Leave one core free
        print(f"Using {n_processes} processes to process {len(args_list)} runs")
        
        with Pool(processes=n_processes) as pool:
            results = pool.map(process_single_run, args_list)
        
        # Filter out None results (failed runs)
        results = [r for r in results if r is not None]
        
        if not results:
            print(f"No valid results for {condition_string}")
            continue
        
        # Aggregate results into arrays
        Nruns = len(results)
        n_agents = params["n_agents"]
        n_beliefs = params["M"]
        n_edges = len(edge_names)
        time_points = results[0]['time_points']
        n_times = len(time_points)
        
        # Initialize aggregated arrays
        metrics = {}
        for key in results[0]['metrics'].keys():
            metrics[key] = np.zeros((Nruns, n_times, n_agents), 
                                    dtype=results[0]['metrics'][key].dtype)
        
        meta_seed = np.zeros(Nruns, dtype=int)
        meta_adaptive = np.zeros(Nruns, dtype=int)
        meta_pressure_strength = np.zeros(Nruns, dtype=int)
        
        if store_beliefAndEdges:
            belief_array = np.zeros((Nruns, n_times, n_agents, n_beliefs), dtype=np.float32)
            edge_array = np.zeros((Nruns, n_times, n_agents, n_edges), dtype=np.float32)
        
        # Aggregate
        for run_idx, result in enumerate(results):
            meta_seed[run_idx] = result['meta']['seed']
            meta_adaptive[run_idx] = result['meta']['adaptive']
            meta_pressure_strength[run_idx] = result['meta']['pressure_strength']
            
            for key in metrics.keys():
                metrics[key][run_idx] = result['metrics'][key]
            
            if store_beliefAndEdges:
                belief_array[run_idx] = result['beliefs']
                edge_array[run_idx] = result['edges']
        
        # Build xarray Dataset
        allData = {key: (("run", "time", "agent_id"), arr) for key, arr in metrics.items()}
        
        coords = {
            "run": np.arange(Nruns),
            "time": time_points,
            "agent_id": range(n_agents),
            "seed": ("run", meta_seed),
            "adaptive": ("run", meta_adaptive),
            "s_ext": ("run", meta_pressure_strength),
        }
        
        if store_beliefAndEdges:
            allData["belief_value"] = (("run", "time", "agent_id", "belief"), belief_array)
            allData["edge_weight"] = (("run", "time", "agent_id", "edge"), edge_array)
            coords["belief"] = params["belief_dimensions"]
            coords["edge"] = params["edgeNames"]
        
        ds = xr.Dataset(allData, coords=coords)
        ds = ds.set_index(run=["adaptive", "seed", "s_ext"])
        ds = ds.unstack("run")
        
        ds.attrs.update({
            "epsilon": [eps for eps, lam in fixadaptive],
            "mu": 0,
            "lambda": [lam for eps, lam in fixadaptive],
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
        })
        
        # Compute response types
        t_beforeEvent = 94.5
        t_inEvent = 144.5
        t_postEvent_short = 194.5
        eval_time = t_postEvent_short
        
        beliefs_beforeEvent = ds.focal_belief.sel(time=t_beforeEvent)
        beliefs_inEvent = ds.focal_belief.sel(time=t_inEvent)
        beliefs_eval = ds.focal_belief.sel(time=eval_time)
        
        responses = [
            "persistent-positive", 
            "non-persistent-positive", 
            "compliant",
            "late-compliant", 
            "resilient", 
            "resistant"
        ]
        response_map = {r: n for n, r in enumerate(responses)}
        response_map["NA"] = 99
        #response_map["unknown"] =    97
        
        conditions = [
            (beliefs_beforeEvent > tresh) & (beliefs_inEvent > tresh) & (beliefs_eval > tresh), # persistent-positive
            (beliefs_beforeEvent > tresh) & (beliefs_inEvent > tresh) & (beliefs_eval < -tresh), # non-persistent-positive
            (beliefs_beforeEvent < -tresh) & (beliefs_inEvent > tresh) & (beliefs_eval > tresh), # compliant
            (beliefs_beforeEvent < -tresh) & (beliefs_inEvent < -tresh) & (beliefs_eval > tresh), # late-compliant
            (beliefs_beforeEvent < -tresh) & (beliefs_inEvent > tresh) & (beliefs_eval < -tresh),  # resilient
            (beliefs_beforeEvent < -tresh) & (beliefs_inEvent < -tresh) & (beliefs_eval < -tresh),  # resistant
            ]
        choices = [response_map[r] for r in responses]
        response = xr.DataArray(
            np.select(conditions, choices, default=99),
            dims=beliefs_beforeEvent.dims,
            coords=beliefs_beforeEvent.coords
        )
        # response = xr.where(
        #     ,
        #     response_map["late-compliant"],
        #     xr.where(
        #         (beliefs_beforeEvent < -tresh) & (beliefs_inEvent < -tresh) & (beliefs_eval < -tresh),
        #         response_map["resistant"],
        #         xr.where(
        #             (beliefs_beforeEvent < -tresh) & (beliefs_inEvent > tresh) & (beliefs_eval < -tresh),
        #             response_map["resilient"],
        #             xr.where(
        #                 (beliefs_beforeEvent < -tresh) & (beliefs_inEvent > tresh) & (beliefs_eval > tresh),
        #                 response_map["compliant"],
        #                 xr.where(
        #                     (beliefs_beforeEvent > tresh) & (beliefs_eval > tresh),
        #                     response_map["persistent-positive"],
        #                     xr.where(
        #                         (beliefs_beforeEvent > tresh) & (beliefs_eval < -tresh),
        #                         response_map["non-persistent-positive"],
        #                         np.nan,
        #                     ),
        #                 ),
        #             ),
        #         ),
        #     ),
        # )
        
        ds.attrs["evaluation_time_for_response"] = eval_time
        ds["response_type"] = response.astype(np.uint8)
        if 0 in pressures.values():
            ds["response_type"].loc[dict(s_ext=0)] = response_map["NA"]
        
        # Save
        ds["s_ext"] = ds["s_ext"].astype(np.uint8)
        ds["time"] = ds["time"].astype(np.float32)
        ds["seed"] = ds["seed"].astype(np.uint16)
        ds["agent_id"] = ds["agent_id"].astype(np.uint8)
        
        fname = f"processed_data/2026-01-21_modelAdaptiveBN_{condition_string}_results_metricsOnly_tresh{tresh}.ncdf"
        ds.to_netcdf(fname)
        print(f"Stored as {fname}")
# %%
