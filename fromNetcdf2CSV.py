import xarray as xr

ds = xr.load_dataset("processed_data/2025-12-16_modelAdaptiveBN_results_metricsOnly.ncdf")

#################################
#####  BELIEF AND EDGE   #####
#################################

dim = "0"
# for belief data in run with adaptive BN (adaptive=True), seed=1, external strength = 0 (or 1,2,4,8,16), the belief 0 (focal) [alternative "1", "2", ...]:
df = (
    ds.sel(adaptive=True, seed=1, s_ext=0, belief=dim)["belief_value"]
    .to_dataframe()
    .reset_index()
)


edgeName = "0_1"  # format: beliefDim1_beliefDim2
# for edge data in run with adaptive BN (adaptive=True), seed=1, external strength = 0, the edge "0_1":
df = (
    ds.sel(adaptive=True, seed=1, s_ext=0, edge=edgeName)["edge_weight"]
    .to_dataframe()
    .reset_index()
)

#################################
#####  RESPONSE_TYPE   #####
#################################

# for response_types in run with adaptive BN (adaptive=True), seed=1, external strength = 1:
# NOTE: response for external_strength=0 is set to Nan (or 99)
#
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
response_map_inv = {n: r for r, n in response_map.items()}
response_map_inv[99] = "NA"
df = (
    ds.sel(adaptive=True, seed=1, s_ext=1)["response_type"]
    .to_dataframe()
    .replace(response_map_inv)
    .reset_index()
)
