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

plt.rcParams.update({"font.size": 10})
# plt.rcParams.update({"font.size":10})


# %%

resultsfolder = "2025-08_results_HenrikMeeting/"

T = 200
params = {
    # General Setup
    "n": 100,
    "belief_options": np.linspace(-1, 1, 7),
    "social_edge_weight": 2.0,
    "memory": 3,
    "M": 10,  # number of beliefs
    "focal_att": "a",
    # Init
    "initial_w": None,
    # Edge Dynamics
    "epsV": None,
    "mu": None,
    "lam": None,
    # Social Network
    "clusters": ["A", "B"],
    "withinCluster_link_prob": 0.4,
    "betweenCluster_link_prob": 0.01,
    # Belief Dynamics
    # "beta_pers": None,
    # "beta_soc": None,
    # Simulation setup:
    "seed": None,
    "T": T,
    "dt": 1,
    "track_times": np.arange(0, T + 1, 1),
    "intervention_period": [],  # range(100, 150),
    "intervention_att": None,  # "b",
    "intervention_strength": None,  # 10,  # 10,
    "intervention_val": None,  # 1,  # 1,
    # "socNetType":"observe-neighbours",  # observe-neighbours or observe-all
    # "socInfType":None,   # correlation or co-occurence or copy
}
params["atts"] = list(string.ascii_lowercase[: params["M"]])
params["edge_list"] = list(combinations(params["atts"], 2))
params["edgeNames"] = [f"({i},{j})" for i, j in params["edge_list"]]

atts = params["atts"]
edge_list = params["edge_list"]
edge_labels = [f"({i},{j})" for i, j in edge_list]
focal_att = params["focal_att"]


def generate_filename(params):
    """Generate filename for results."""
    social_net = f"(Stoch-{len(params['clusters'])}-Block-{params['withinCluster_link_prob']}-{params['betweenCluster_link_prob']})"
    intervention = (
        f"_noIntervention"
        if params["intervention_att"] is None
        else f"_interv{params['intervention_period'][0]}-{params['intervention_period'][-1]}-{params['intervention_att']}-strength{params['intervention_strength']}-value{params['intervention_val']}"
    )
    return (
        f"{resultsfolder}adaptiveBN_M-{params['M']}_n-{params['n']}-{social_net}"
        f"_epsV{params['epsV']}-m{params['memory']}_mu{params['mu']}"
        f"_lam{params['lam']}_rho{params['social_edge_weight']}_initialW-{params['initial_w']}"
        f"{intervention}"
        f"_seed{params['seed']}"
    ), intervention


# %%
seeds = range(100)
epsV = 0.0
mu = 0.0
paramCombis = [
    #  (init_w, epsV, mu, seed)
    (initial_w, epsV, mu, s)
    for initial_w in [0.2, 0.8]
    for epsV, mu in [(0.0, 0.0), (0.5, 0.5)]
    for s in seeds
]

resNodes = []
resEdges = []

all_frob_dists = {}
for initial_w, epsV, mu, seed in paramCombis:
    if seed == 0:
        print(initial_w, epsV, mu, seed)
    params["eps"] = 0.0
    params["lam"] = 0.0
    params["epsV"] = epsV
    params["mu"] = mu
    params["initial_w"] = initial_w
    params["seed"] = seed

    simOut = pd.read_csv(generate_filename(params)[0] + ".csv", low_memory=False)

    neighbours_dict = {
        ag: json.loads(
            simOut.loc[
                (simOut.time == 0) & (simOut.agent_id == ag), "neighbours"
            ].values[0]
        )
        for ag in range(0, params["n"])
    }
    G = nx.from_dict_of_lists(neighbours_dict)

    std = simOut.loc[simOut.time == T, focal_att].std()
    extreme = simOut.loc[simOut.time == T, focal_att].abs().mean()

    std_A = simOut.loc[(simOut.time == T) & (simOut.identity == "A"), focal_att].std()

    std_B = simOut.loc[(simOut.time == T) & (simOut.identity == "B"), focal_att].std()

    delBelief_AB = np.abs(
        simOut.loc[(simOut.time == T) & (simOut.identity == "A"), focal_att].mean()
        - simOut.loc[(simOut.time == T) & (simOut.identity == "B"), focal_att].mean()
    )

    ags_with_signed_ops = [
        ag
        for ag in G.nodes()
        if simOut.loc[(simOut.time == T) & (simOut.agent_id == ag), focal_att].values[0]
        != 0
    ]
    G = nx.subgraph(G, ags_with_signed_ops)
    communities = [
        simOut.loc[(simOut.time == T) & (np.sign(simOut[focal_att]) == sign)][
            "agent_id"
        ].values
        for sign in [-1, 1]
    ]
    modularity = nx.community.modularity(
        G,
        communities,
    )

    if max(np.diff(simOut.time.unique())) == 1:
        stationarity = simOut.groupby("agent_id")[focal_att].diff().abs().sum() / len(
            simOut["agent_id"].unique()
        )
    else:
        stationarity = np.nan

    resNodes.append(
        [
            initial_w,
            epsV,
            mu,
            seed,
            focal_att,
            std,
            extreme,
            std_A,
            std_B,
            delBelief_AB,
            modularity,
            stationarity,
        ]
    )

    #################################
    #####  EDGES   #####
    #################################
    std_edges = simOut.loc[simOut.time == T, edge_labels].std().mean()

    extremeness_edges = simOut.loc[simOut.time == T, edge_labels].abs().mean().mean()

    std_edges_A = (
        simOut.loc[(simOut.time == T) & (simOut.identity == "A"), edge_labels]
        .std(axis=0)
        .mean()
    )
    std_edges_B = (
        simOut.loc[(simOut.time == T) & (simOut.identity == "B"), edge_labels]
        .std(axis=0)
        .mean()
    )
    std_edges_within = np.mean([std_edges_A, std_edges_B])

    dEdges_AB_mean = (
        (
            simOut.loc[(simOut.time == T) & (simOut.identity == "A"), edge_labels].mean(
                axis=0
            )
            - simOut.loc[
                (simOut.time == T) & (simOut.identity == "B"), edge_labels
            ].mean(axis=0)
        )
        .abs()
        .mean()
    )

    stdEdges_perAgent_mean = np.mean(
        [
            simOut.loc[(simOut.time == T) & (simOut.agent_id == i)][edge_labels].std(
                axis=1
            )
            for i in simOut.agent_id.unique()
        ]
    )

    if max(np.diff(simOut.time.unique())) == 1:
        edge_stationarity = (
            simOut.groupby("agent_id")[edge_labels].diff().abs().sum().mean()
            / len(edge_labels)
            / len(simOut["agent_id"].unique())
        )
    else:
        edge_stationarity = np.nan

    # Filter snapshot at time T
    df = simOut.loc[simOut.time == T, edge_labels + ["agent_id", "identity"]]
    # Compute pairwise Euclidean distances
    dist_array = pdist(df[edge_labels], metric="euclidean")
    dist_matrix = pd.DataFrame(
        squareform(dist_array), index=df.agent_id.values, columns=df.agent_id.values
    )
    # Keep only lower triangle (excluding diagonal)
    lower_triangle = dist_matrix.where(
        np.tril(np.ones(dist_matrix.shape), k=-1).astype(bool)
    )
    lower_triangle = lower_triangle.reset_index().rename(columns={"index": "ag1"})
    # Melt to long format
    dists = lower_triangle.melt(
        id_vars="ag1", var_name="ag2", value_name="Frobenius_distance"
    ).dropna()
    all_frob_dists[f"initW={initial_w}_epsV{epsV}_mu{mu}_seed{seed}"] = dists
    # Map agent_id to identity
    id_map = df.set_index("agent_id")["identity"]
    # Create within/between identity interaction types
    dists["groups"] = dists.apply(
        lambda x: "-".join(sorted([id_map[x.ag1], id_map[x.ag2]])), axis=1
    )
    dists["within"] = (dists.groups == "A-A") | (dists.groups == "B-B")
    avgFrob = dists.groupby("within")["Frobenius_distance"].mean().reset_index()

    resEdges.append(
        [initial_w, epsV, mu, seed]
        + [
            std_edges,
            extremeness_edges,
            std_edges_within,
            dEdges_AB_mean,
            stdEdges_perAgent_mean,
            edge_stationarity,
            avgFrob.loc[~avgFrob.within, "Frobenius_distance"].values[0],
            avgFrob.loc[avgFrob.within, "Frobenius_distance"].values[0],
        ]
    )
# %%
outcomeColsNodes = [
    "std",
    "extreme",
    "std_A",
    "std_B",
    "delBelief_AB",
    "modularity",
    "stationarity",
]

outcomeColsEdges = [
    "std_edges",
    "extremeness_edges",
    "std_edges_within",
    "dEdges_AB_mean",
    "stdEdges_\nperAgent_mean",
    "edge_stationarity",
    "FrobDist_between-group",
    "FrobDist_within-group",
]


inputCols = ["initial_w", "epsV", "mu", "seed", "attitude"]
resNodes_df = pd.DataFrame(resNodes, columns=inputCols + outcomeColsNodes)
inputCols = ["initial_w", "epsV", "mu", "seed"]
resEdges_df = pd.DataFrame(resEdges, columns=inputCols + outcomeColsEdges)
for res_df in [resNodes_df, resEdges_df]:
    res_df["input"] = res_df.apply(
        lambda x: rf"$\omega_0={x['initial_w']}$, "
        + ("fixed" if x["epsV"] == 0.0 else "adaptive"),
        # + "\n"
        # + rf"$\epsilon={x['epsV']},$ $\mu={x['mu']}$"
        # + (f", {x['attitude']}" if "attitude" in res_df.columns else ""),
        axis=1,
    )

# %%
#################################
#####  PLot Node Outcomes   #####
#################################

# fig, axs = plt.subplots(
#     2, int((len(outcomeColsNodes) + 1) / 2), figsize=(16, 14), sharex=False
# )
# fig.suptitle("BELIEF CHANGE (NODE)")
# resNodes_df = resNodes_df.sort_values("initial_w")
# hue_order = resNodes_df["input"].unique()
# palette = sns.color_palette("Paired", n_colors=8)
# color_dict = dict(zip(hue_order, palette))

# for i, col in enumerate(outcomeColsNodes):
#     ax = axs.flatten()[i]
#     sns.boxplot(
#         y=col,
#         data=resNodes_df,
#         ax=ax,
#         hue="input",
#         whis=[5, 95],
#         fliersize=0,
#         dodge=True,
#         palette=color_dict,
#     )
#     # Recolor boxplot lines manually (whiskers, caps, medians)
#     n_hues = len(hue_order)
#     lines = ax.lines
#     for j in range(n_hues):
#         color = color_dict[hue_order[j]]
#         # Each hue group has 6 lines: whiskers(2), caps(2), median(1), box outline (1) — may vary by seaborn version
#         for k in range(6):
#             line_idx = j * 6 + k
#             if line_idx < len(lines) and not k == 4:
#                 lines[line_idx].set_color(color)

#     sns.stripplot(
#         y=col,
#         data=resNodes_df,
#         ax=ax,
#         hue="input",
#         dodge=True,
#         jitter=True,
#         size=8,
#         alpha=0.6,
#         edgecolor="auto",
#         linewidth=0.3,
#         legend=False,
#         palette=palette,
#     )
#     ax.set_ylabel("")
#     ax.set_title(col)

#     # Remove the second legend from stripplot to avoid duplicates
#     if ax.get_legend():
#         ax.get_legend().remove()

# axs.flatten()[-1].axis(False)
# # Create a single legend for the entire figure
# handles, labels = axs.flatten()[-2].get_legend_handles_labels()
# fig.legend(
#     handles,
#     labels,
#     loc="upper left",
#     ncol=1,
#     bbox_to_anchor=(0.75, 0.48),
#     labelspacing=1.5,
# )

# fig.tight_layout(w_pad=0.0)
# # axs.flatten()[outcomeCols.index("modularity")].text(
# #     0.0, 0.00, "not implemented", ha="center", va="center", rotation=90
# # )

# %%

#################################
#####  COMBI    #####
#################################
fig, axs = plt.subplots(2, 4, figsize=(18 / 2.54, 14 / 2.54), sharex=False)
resNodes_df = resNodes_df.sort_values("initial_w")
hue_order = resNodes_df["input"].unique()
palette = sns.color_palette("Paired", n_colors=4)
color_dict = dict(zip(hue_order, palette))

outs = ["std", "extreme", "modularity"]
outsE = [
    "std_edges",
    "extremeness_edges",
    "FrobDist_between-group",
    "FrobDist_within-group",
]
for i, col in enumerate(outs):
    ax = axs[0, i]
    sns.boxplot(
        y=col,
        data=resNodes_df,
        ax=ax,
        hue="input",
        whis=[5, 95],
        fliersize=0,
        dodge=True,
        palette=color_dict,
    )
    # Recolor boxplot lines manually (whiskers, caps, medians)
    n_hues = len(hue_order)
    lines = ax.lines
    for j in range(n_hues):
        color = color_dict[hue_order[j]]
        # Each hue group has 6 lines: whiskers(2), caps(2), median(1), box outline (1) — may vary by seaborn version
        for k in range(6):
            line_idx = j * 6 + k
            if line_idx < len(lines) and not k == 4:
                lines[line_idx].set_color(color)

    sns.stripplot(
        y=col,
        data=resNodes_df,
        ax=ax,
        hue="input",
        dodge=True,
        jitter=True,
        size=2,
        alpha=0.6,
        edgecolor="auto",
        linewidth=0.3,
        legend=False,
        palette=palette,
    )
    ax.set_ylabel("")
    ax.set_title(col)

    if "extreme" in col:
        ax.set_ylim(0, 1)
    if "std" in col or "modularity" in col:
        ax.set_ylim(
            0,
        )

    # Remove the second legend from stripplot to avoid duplicates
    if ax.get_legend():
        ax.get_legend().remove()
    ax.set_xticks([])


for i, col in enumerate(outsE):
    ax = axs[1, i]
    sns.boxplot(
        y=col,
        data=resEdges_df,
        hue_order=resEdges_df["input"].unique(),
        ax=ax,
        hue="input",
        whis=[5, 95],
        fliersize=0,
        dodge=True,
        palette=color_dict,
    )
    # Recolor boxplot lines manually (whiskers, caps, medians)
    n_hues = len(hue_order)
    lines = ax.lines
    for j in range(n_hues):
        color = color_dict[hue_order[j]]
        # Each hue group has 6 lines: whiskers(2), caps(2), median(1), box outline (1) — may vary by seaborn version
        for k in range(6):
            line_idx = j * 6 + k
            if line_idx < len(lines) and not k == 4:
                lines[line_idx].set_color(color)
    sns.stripplot(
        y=col,
        data=resEdges_df,
        ax=ax,
        hue="input",
        dodge=True,
        jitter=True,
        size=2,
        alpha=0.6,
        edgecolor="auto",
        linewidth=0.3,
        legend=False,
        palette=palette,
    )

    ax.set_ylabel("")
    ax.set_title(col)
    if "FrobDist" in col:
        ax.set_ylim(0, 8)
    if "extremeness" in col:
        ax.set_ylim(
            0,
        )
    if "std" in col or "modularity" in col:
        ax.set_ylim(
            0,
        )

    # Remove the second legend from stripplot to avoid duplicates
    if ax.get_legend():
        ax.get_legend().remove()
    ax.set_xticks([])

axs[0, -1].axis(False)
# Create a single legend for the entire figure
handles, labels = axs[0, -2].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper left",
    ncol=1,
    bbox_to_anchor=(0.75, 0.88),
    labelspacing=1.5,
)

axs[0, 0].set_ylabel("FOCAL BELIEF", fontsize=12)
axs[1, 0].set_ylabel("EDGES", fontsize=12)
axs[0, 0].set_title("std", fontsize=9)
axs[0, 1].set_title("extremeness", fontsize=9)
axs[0, 2].set_title("modularity", fontsize=9)
axs[1, 0].set_title("std", fontsize=9)
axs[1, 1].set_title("extremeness", fontsize=9)
axs[1, 2].set_title("Frobenius distance\n(between groups)", fontsize=9)
axs[1, 3].set_title("Frobenius distance\n(within groups)", fontsize=9)
fig.tight_layout()
plt.savefig(
    resultsfolder
    + "figs/"
    + f"outcomes_epsV{epsV}_mu{mu}_initW{initial_w}_t{T}_rho{params['social_edge_weight']}{generate_filename(params)[1]}.pdf"
)

# %%
# #####  plot edge outcomes   #####
# #################################
# fig, axs = plt.subplots(
#     2, int((len(outcomeColsEdges) + 1) / 2) + 1, figsize=(16, 14), sharex=False
# )
# fig.suptitle("BELIEF NETWORK CHANGE (EDGE)")
# hue_order = resEdges_df["input"].unique()
# palette = sns.color_palette("Paired", n_colors=8)
# color_dict = dict(zip(hue_order, palette))

# for i, col in enumerate(outcomeColsEdges):
#     ax = axs.flatten()[i]
#     sns.boxplot(
#         y=col,
#         data=resEdges_df,
#         hue_order=resEdges_df["input"].unique(),
#         ax=ax,
#         hue="input",
#         whis=[5, 95],
#         fliersize=0,
#         dodge=True,
#         palette=color_dict,
#     )
#     # Recolor boxplot lines manually (whiskers, caps, medians)
#     n_hues = len(hue_order)
#     lines = ax.lines
#     for j in range(n_hues):
#         color = color_dict[hue_order[j]]
#         # Each hue group has 6 lines: whiskers(2), caps(2), median(1), box outline (1) — may vary by seaborn version
#         for k in range(6):
#             line_idx = j * 6 + k
#             if line_idx < len(lines) and not k == 4:
#                 lines[line_idx].set_color(color)
#     sns.stripplot(
#         y=col,
#         data=resEdges_df,
#         ax=ax,
#         hue="input",
#         dodge=True,
#         jitter=True,
#         size=8,
#         alpha=0.6,
#         edgecolor="auto",
#         linewidth=0.3,
#         legend=False,
#         palette=palette,
#     )

#     ax.set_ylabel("")
#     ax.set_title(col)
#     if "FrobDist" in col:
#         ax.set_ylim(0, 5)

#     # Remove the second legend from stripplot to avoid duplicates
#     if ax.get_legend():
#         ax.get_legend().remove()

# axs.flatten()[-1].axis(False)
# axs.flatten()[-2].axis(False)
# # Create a single legend for the entire figure
# handles, labels = axs.flatten()[-3].get_legend_handles_labels()
# fig.legend(
#     handles,
#     labels,
#     loc="upper left",
#     ncol=1,
#     bbox_to_anchor=(0.75, 0.48),
#     labelspacing=1.5,
# )


# fig.tight_layout(w_pad=0.1)


# %%
#####  PLOT TIME SERIES   #####
#################################


def plot_net(simOut, t, params, pos=None, ax=ax, cbar=True):
    neighbours_dict = {
        ag: json.loads(
            simOut.loc[
                (simOut.time == 0) & (simOut.agent_id == ag), "neighbours"
            ].values[0]
        )
        for ag in range(0, params["n"])
    }
    G = nx.from_dict_of_lists(neighbours_dict)
    colors = [
        (simOut.loc[(simOut.time == t) & (simOut.agent_id == i), focal_att])
        for i in neighbours_dict.keys()
    ]
    pos = nx.spring_layout(G, seed=21)  # if t > 0 else pos
    nx.draw_networkx_edges(G, pos=pos, width=0.5, alpha=0.5, ax=ax)
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        node_color=colors,
        cmap="coolwarm",
        vmax=1,
        vmin=-1,
        node_size=10,
        ax=ax,
    )
    # ax.set_title(
    #     # rf"$\beta_p = {params['beta_pers']}, \beta_s={params['beta_soc']},
    #     rf"$\omega_0 = {params['initial_w']}, \epsilon={params['epsV']}, \mu={params['mu']}$, seed={params['seed']}"
    # )
    if cbar:
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1))
        sm._A = []
        plt.colorbar(
            sm,
            label="focal belief",
            ticks=[-1, 0, 1],
            ax=ax,
            orientation="horizontal",
            pad=0.05,
        )
    ax.text(
        0.05,
        0.05,
        f"$t={t}$",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=10,
        color="grey",
    )
    return ax, pos


s = 3
initial_w, epsV, mu, seed = (0.2, 0.5, 0.5, s)
params["initial_w"], params["epsV"], params["mu"], params["seed"] = (
    initial_w,
    epsV,
    mu,
    seed,
)
simOut2 = pd.read_csv(generate_filename(params)[0] + ".csv", low_memory=False)

initial_w, epsV, mu, seed = (0.8, 0.5, 0.5, s)
params["initial_w"], params["epsV"], params["mu"], params["seed"] = (
    initial_w,
    epsV,
    mu,
    seed,
)
simOut8 = pd.read_csv(generate_filename(params)[0] + ".csv", low_memory=False)

# simOut.loc[(simOut.agent_id == 0), focal_att]


fig, axs = plt.subplots(
    2, 4, figsize=(18 / 2.54, 12 / 2.54), width_ratios=[2, 4, 2, 4], sharex="col"
)

params["initial_w"], params["epsV"], params["mu"], params["seed"] = (
    0.2,
    epsV,
    mu,
    seed,
)
plot_net(simOut2, 0, params, ax=axs[0, 0], cbar=False)
axs[0, 0].set_ylabel(rf"$\omega_0 = {params['initial_w']}$")
axs[0, 0].set_title("social network\n($t=0$)", fontsize=9)
params["initial_w"] = 0.8
plot_net(simOut8, 0, params, ax=axs[1, 0])
axs[1, 0].set_ylabel(rf"$\omega_0 = {params['initial_w']}$")


ax = axs[0, 1]
inds = [2, 12, 22, 32, 42, 52, 62, 72, 82, 92]
for i in inds:
    simOut2.loc[(simOut2.agent_id == i), ["time", focal_att]].plot(
        x="time", ax=ax, legend=False, lw=1, alpha=0.5
    )
ax.set_xlim(0, T)
ax.set_title("focal belief over $t$\n(for 10 agents)", fontsize=9)
ax.set_yticks([-1, 0, 1])
ax.set_ylabel("focal belief", labelpad=-7)

ax = axs[1, 1]
for i in inds:
    simOut8.loc[(simOut8.agent_id == i), ["time", focal_att]].plot(
        x="time", ax=ax, legend=False, lw=1, alpha=0.5
    )
ax.set_xlim(0, T)
ax.set_title("")
ax.set_yticks([-1, 0, 1])
ax.set_ylabel("focal belief", labelpad=-7)

plot_net(simOut2, T, params, ax=axs[0, 2], cbar=False)
axs[0, 2].set_title("social network\n" + rf"($t={T}$)", fontsize=9)
params["initial_w"] = 0.8
plot_net(simOut8, T, params, ax=axs[1, 2], cbar=False)


for ax_kde, simOut in zip([axs[0, 3], axs[1, 3]], [simOut2, simOut8]):
    # Filter snapshot at time T
    df = simOut.loc[simOut.time == T, edge_labels + ["agent_id", "identity"]]
    # Compute pairwise Euclidean distances
    dist_array = pdist(df[edge_labels], metric="euclidean")
    dist_matrix = pd.DataFrame(
        squareform(dist_array), index=df.agent_id.values, columns=df.agent_id.values
    )
    # Keep only lower triangle (excluding diagonal)
    lower_triangle = dist_matrix.where(
        np.tril(np.ones(dist_matrix.shape), k=-1).astype(bool)
    )
    lower_triangle = lower_triangle.reset_index().rename(columns={"index": "ag1"})
    # Melt to long format
    dists = lower_triangle.melt(
        id_vars="ag1", var_name="ag2", value_name="Frobenius_distance"
    ).dropna()
    # Map agent_id to identity
    id_map = df.set_index("agent_id")["identity"]
    # Create within/between identity interaction types
    dists["groups"] = dists.apply(
        lambda x: "-".join(sorted([id_map[x.ag1], id_map[x.ag2]])), axis=1
    )
    dists["groups"] = dists["groups"].map(
        {"A-A": "G1-G1", "B-B": "G2-G2", "A-B": "G1-G2"}
    )
    dists["within"] = (dists.groups == "G1-G1") | (dists.groups == "G2-G2")
    avgFrob = dists.groupby("within")["Frobenius_distance"].mean().reset_index()

    sns.histplot(
        data=dists,
        x="Frobenius_distance",
        hue="groups",
        ax=ax_kde,
        alpha=0.2,
        bins=np.arange(0, 8, 0.2),
        # element="step",
        stat="count",
        legend=True if ax_kde == axs[1, 3] else False,
        #    common_norm=False
    )
    ax_kde.set_ylim(0, min(ax_kde.get_ylim()[1], 100 * (100 - 1) / 2 / 2))
    ax_kde.set_ylabel("density")
    ax_kde.set_yticks([])
    ax_kde.set_xlim(
        0,
    )
axs[1, 3].set_xlabel(
    "Frobenius distance " + r"of $\vec{\omega}^A$ and $\vec{\omega}^B$"
)
axs[0, 3].set_title("Pairwise comparison of\nagents' belief networks", fontsize=9)

for ax, l in zip(axs.flatten(), string.ascii_uppercase):
    ax.text(0.0, 1.0, l, transform=ax.transAxes, va="top", weight="bold", fontsize=12)

plt.tight_layout()
plt.savefig(resultsfolder + "figs/" + "dynamics_adaptiveNB.pdf")

# %%
#################################
#####  Time series   #####
#################################


def plot_timeSeries_singleAgent(i, s, initial_w, epsV, mu):
    params.update(
        {
            "initial_w": initial_w,
            "epsV": epsV,
            "mu": mu,
            "seed": s,
        }
    )
    simOut = pd.read_csv(generate_filename(params)[0] + ".csv", low_memory=False)
    fig, axs = plt.subplots(2, 1, sharex=True)

    edgeNames = [f"({a},{b})" for a, b in list(combinations(atts, 2))]

    axs[0].plot([], [], color="k", label="focal belief")
    axs[0].plot([], [], color="white", label="non-focal beliefs")
    axs[0].legend()
    axs[1].plot([], [], color="k", label="edge: focal & non-focal beliefs")
    axs[1].plot([], [], color="r", label="edge: two non-focal beliefs")
    axs[1].legend()

    simOut.loc[(simOut.agent_id == i), ["time"] + atts].drop(columns=focal_att).rolling(
        10
    ).mean().plot(x="time", ax=axs[0], legend=False, alpha=0.5, label="_nolegend_")
    simOut.loc[(simOut.agent_id == i), ["time"] + [focal_att]].rolling(10).mean().plot(
        x="time",
        ax=axs[0],
        legend=False,
        lw=2,
        color="k",
        alpha=0.5,
        label="_nolegend_",
    )

    simOut.loc[
        (simOut.agent_id == i), ["time"] + [e for e in edgeNames if focal_att not in e]
    ].rolling(10).mean().plot(
        x="time", ax=axs[1], legend=False, color="red", alpha=0.5, label="_nolegend_"
    )
    simOut.loc[
        (simOut.agent_id == i), ["time"] + [e for e in edgeNames if focal_att in e]
    ].rolling(10).mean().plot(
        x="time", ax=axs[1], legend=False, color="k", alpha=0.5, label="_nolegend_"
    )

    if len(params["intervention_period"]) > 0:
        axs[0].fill_between(
            [params["intervention_period"][0], params["intervention_period"][-1]],
            [-1, -1],
            [1, 1],
            zorder=-2,
            color="gainsboro",
            alpha=1,
        )
        axs[1].fill_between(
            [params["intervention_period"][0], params["intervention_period"][-1]],
            [-1, -1],
            [1, 1],
            zorder=-2,
            color="gainsboro",
            alpha=1,
        )
    axs[1].hlines(0, T, 0, linestyles="--", zorder=-1, color="grey")
    axs[0].hlines(0, T, 0, linestyles="--", zorder=-1, color="grey")
    axs[0].set_ylim(-1, 1)
    axs[0].set_ylabel("beliefs")
    axs[1].set_ylabel("BN edge weights")
    axs[1].text(
        0.99, 0.01, "rolling mean", va="bottom", ha="right", transform=axs[1].transAxes
    )
    fig.suptitle(
        rf"Agent {i} in simulation {seed} ($\omega_0={initial_w}$, $\epsilon_V={epsV}$, $\mu={mu}$)"
    )
    return fig, ax


# %%
for i in [44, 97]:
    plot_timeSeries_singleAgent(i=i, s=11, initial_w=0.8, epsV=0.0, mu=0.0)

# %%


#################################
#####  VIS   #####
#################################


# %%
# epsV = 1.0
# mu = 0.5
# paramCombis = [
#     #  (beta_pers, beta_soc, epsV, mu, seed)
#     (2, 2, epsV, mu, s)
#     for s in range(20)
# ]


# belief_observe = "avgbelief"


# res_arr = []
# for beta_pers, beta_soc, epsV, mu, seed in paramCombis:
#     eps = 0.0
#     lam = 0.0
#     params["initial_w"], params["epsV"], params["mu"], params["seed"] = (
#         0.2,
#         epsV,
#         mu,
#         seed,
#     )
#     filename = generate_filename(params)[0]
#     simOut = pd.read_csv(filename + ".csv")

#     for t in [T]:
#         simOut.loc[simOut.time == t, "avgbelief"] = simOut.loc[
#             simOut.time == t, atts
#         ].mean(axis=1)

#     res_arr.append(simOut.loc[simOut.time == T, belief_observe].values)


# fig = plt.figure()

# ax = fig.add_subplot(111)
# sns.histplot(
#     np.array(res_arr)[np.random.randint(len(res_arr), size=3), :].T,
#     bins=np.linspace(-1 - 1 / 14, 1 + 1 / 14, 21),
#     palette="Set1",
#     alpha=0.2,
#     legend=True,
#     ax=ax,
# )
# leg = ax.get_legend()
# leg.set_title("seed")
# ax.set_title(
#     rf"$\beta_p={beta_pers}, \beta_s={beta_soc}$"
#     + "\n"
#     + rf"$\epsilon_V={epsV}, \mu={mu}$"
# )

#################################
#####  Plot Networks   #####
#################################


# for t in range(100):
#     seed_samples = [0]#np.random.choice(range(10), size=10, replace=False)
#     for s in seed_samples:
#         fig = plt.figure()
#         ax = plt.axs()
#         colors = [(simOut.loc[(simOut.time==t) & (simOut.agent_id==i), focal_att]) for i in neighbours_dict.keys()]
#         pos = nx.spring_layout(G, seed=1) if t>0 else pos
#         nx.draw_networkx_edges(G, pos=pos, width=0.5, alpha=0.5)
#         nx.draw_networkx_nodes(G, pos=pos, node_color=colors, cmap="coolwarm", vmax=1, vmin=-1, node_size=100)
#         ax.set_title(fr"$\beta_p = {beta_pers}, \beta_s={beta_soc}, \epsilon={epsV}, \mu={mu}$")
#         sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin = -1, vmax=1))
#         sm._A = []
#         plt.colorbar(sm, label="focal attitude", ticks=[-1,0,1], ax=ax)
#         ax.annotate("agents", pos[38], (pos[38][0]+0.3, pos[38][1]-0.3), fontsize=15, color="grey",  arrowprops=dict( arrowstyle="-", connectionstyle="arc3,rad=0.2",color="grey", shrinkA=5, shrinkB=5))
#         ax.annotate("", pos[56], (pos[38][0]+0.4, pos[38][1]-0.3), fontsize=15, color="grey", arrowprops=dict( arrowstyle="-", connectionstyle="arc3,rad=-0.2",color="grey", shrinkA=5, shrinkB=5))
#         ax.annotate("", pos[58], (pos[38][0]+0.5, pos[38][1]-0.3), fontsize=15, color="grey", arrowprops=dict( arrowstyle="-", connectionstyle="arc3,rad=-0.2",color="grey", shrinkA=5, shrinkB=5))
#         ax.text(0.05,0.05,f"$t={t}$", transform=ax.transAxes, va="bottom", ha="left", fontsize=15, color="grey")
#         plt.savefig(f"figs_gif_CA-workshop/socNet_epsV-{epsV}-m{params['memory']}_mu{mu}_lam{lam}_initialW-{params['initial_w']}_beta-{beta_pers}-{beta_soc}_seed{s}_t{t:03d}.png")

# # %%
# epsV = 0.3
# mu = 0.5
# s = 0
# initial_w = 0.2


# params["initial_w"], params["epsV"], params["mu"], params["seed"] = (
#     0.2,
#     epsV,
#     mu,
#     seed,
# )
# filename = generate_filename(params)[0]
# simOut = pd.read_csv(filename + ".csv", low_memory=False)


# selected_edgeNames = ["(a,b)", "(d,e)"]
# fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
# sns.histplot(
#     simOut.loc[(simOut.time == T) & (simOut.identity == "A")][selected_edgeNames],
#     legend=False,
#     ax=axs[0],
#     bins=np.linspace(-0.75, 0.75, 12),
# )
# sns.histplot(
#     simOut.loc[(simOut.time == T) & (simOut.identity == "B")][selected_edgeNames],
#     legend=True,
#     ax=axs[1],
#     bins=np.linspace(-0.75, 0.75, 12),
# )
# axs[0].set_title("group A")
# axs[1].set_title("group B")


# fig.suptitle(
#     "edge weights \n"
#     # + rf"$\beta_p = {beta_pers}$, $\beta_s={beta_soc}$"
#     + rf"$\omega_0 = {initial_w}$"
#     + "\n"
#     + rf"and $\epsilon = {epsV}$, $\mu={mu}$ (colour=edge)"
# )
# fig.tight_layout()

# # %%
# edgeNames = [f"({a},{b})" for a, b in list(combinations(atts, 2))]
# diff = simOut.loc[(simOut.time == T) & (simOut.identity == "A"), edgeNames].mean(
#     axis=0
# ) - simOut.loc[(simOut.time == T) & (simOut.identity == "B")][edgeNames].mean(axis=0)

# fig, ax = plt.subplots(1, 1)
# sns.histplot(diff, ax=ax)
# ax.set_xlabel("Group differences in edge weights")
# ax.set_ylabel("number of edges")
# # ax.set_yticks([])


# # %%
# e_A = simOut.loc[(simOut.time == T) & (simOut.identity == "A")][edgeNames]
# e_B = simOut.loc[(simOut.time == T) & (simOut.identity == "B")][edgeNames]
# inds = (e_A.mean() - e_B.mean()).sort_values().index

# e_B_long = e_B.loc[:, inds].melt(
#     var_name="Belief Network Edges", value_name="Edge Weight"
# )
# e_B_long["Group"] = "B"
# e_A_long = e_A.loc[:, inds].melt(
#     var_name="Belief Network Edges", value_name="Edge Weight"
# )
# e_A_long["Group"] = "A"
# df = pd.concat([e_A_long, e_B_long])

# ax = plt.axes()
# sns.boxplot(
#     data=df,
#     ax=ax,
#     x="Belief Network Edges",
#     y="Edge Weight",
#     hue="Group",
#     whis=[0, 100],
#     palette={"A": "purple", "B": "green"},
# )
# ax.legend(title="Group")
# ax.set_xticklabels([])
# plt.tight_layout()


# %%


# %%
def plot_net(simOut, t, params, pos=None, ax=ax, cbar=True):
    neighbours_dict = {
        ag: json.loads(
            simOut.loc[
                (simOut.time == 0) & (simOut.agent_id == ag), "neighbours"
            ].values[0]
        )
        for ag in range(0, params["n"])
    }
    G = nx.from_dict_of_lists(neighbours_dict)
    colors = [
        (simOut.loc[(simOut.time == t) & (simOut.agent_id == i), focal_att].iloc[0])
        for i in neighbours_dict.keys()
    ]
    pos = nx.spring_layout(G, seed=21)  # if t > 0 else pos
    nx.draw_networkx_edges(G, pos=pos, width=0.5, alpha=0.5, ax=ax)
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        node_color=colors,
        cmap="coolwarm",
        vmax=1,
        vmin=-1,
        node_size=3,
        ax=ax,
    )

    if cbar:
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1))
        sm._A = []
        plt.colorbar(
            sm,
            label="focal belief",
            ticks=[-1, 0, 1],
            ax=ax,
            orientation="horizontal",
            pad=0.05,
        )
    mod = resNodes_df.loc[
        (resNodes_df.initial_w == params["initial_w"])
        & (resNodes_df.epsV == params["epsV"])
        & (resNodes_df.mu == params["mu"])
        & (resNodes_df.seed == params["seed"]),
        "modularity",
    ].values[0]
    sm = list(
        plt.cm.ScalarMappable(
            cmap="YlOrRd", norm=plt.Normalize(vmin=0, vmax=0.3)
        ).get_cmap()(mod)
    )
    sm[-1] = 0.4
    ax.set_facecolor(sm)

    return ax, pos


epsV = 0.0
mu = 0.0
initial_w = 0.2
t = 0


def plot_nets_tot(t, epsV, mu, initial_w):
    fig, axs = plt.subplots(4, 5)
    for s, ax in zip(range(100), axs[:, :].flatten()):
        params["initial_w"], params["epsV"], params["mu"], params["seed"] = (
            initial_w,
            epsV,
            mu,
            s,
        )
        filename = generate_filename(params)[0]
        simOut = pd.read_csv(filename + ".csv", low_memory=False)
        plot_net(simOut, t, params, ax=ax, cbar=(s == 17))

    fig.suptitle(
        rf"focal beliefs at t={t}, $\omega_0={initial_w}$, $\epsilon={epsV}$, $\mu={mu}$, $\rho={params['social_edge_weight']}$"
    )

    fig.tight_layout()
    fig.text(
        0.98,
        0.02,
        "background color =\n modularity of focal beliefs",
        fontsize=7,
        ha="right",
        va="bottom",
        transform=fig.transFigure,
    )


epsV = 0.5
mu = 0.5
ps = [
    (0.2, 0.0, 0.0),
    (0.2, epsV, mu),
    (0.8, 0.0, 0.0),
    (0.8, epsV, mu),
]

for initial_w, epsV, mu in ps:
    t = T
    plot_nets_tot(t, epsV, mu, initial_w)
    plt.savefig(
        resultsfolder
        + "figs/"
        + f"focalNets_epsV{epsV}_mu{mu}_initW{initial_w}_t{t}_rho{params['social_edge_weight']}{generate_filename(params)[1]}.pdf",
        bbox_inches="tight",
    )
# %%


sns.histplot(resNodes_df, x="modularity", hue="input", kde=True, alpha=0.2)
# %%

for initial_w in [0.2, 0.8]:
    epsV = 0.3
    mu = 0.5
    fig, axs = plt.subplots(
        4, 5, sharex=True, sharey=True, figsize=(16 / 2.54, 16 / 2.54)
    )
    for s, ax in zip(range(20), axs[:, :].flatten()):
        params["initial_w"], params["epsV"], params["mu"], params["seed"] = (
            initial_w,
            epsV,
            mu,
            s,
        )
        filename = generate_filename(params)[0]
        simOut = pd.read_csv(filename + ".csv", low_memory=False)
        if s == 0:
            print(initial_w, epsV, mu, s)
        params["eps"] = 0.0
        params["lam"] = 0.0
        params["epsV"] = epsV
        params["mu"] = mu
        params["initial_w"] = initial_w
        params["seed"] = s

        simOut = pd.read_csv(generate_filename(params)[0] + ".csv", low_memory=False)

        neighbours_dict = {
            ag: json.loads(
                simOut.loc[
                    (simOut.time == 0) & (simOut.agent_id == ag), "neighbours"
                ].values[0]
            )
            for ag in range(0, params["n"])
        }
        G = nx.Graph()
        G.add_nodes_from(list(neighbours_dict.keys()))

        dists = all_frob_dists[f"initW={initial_w}_epsV{epsV}_mu{mu}_seed{s}"]
        edge_list = [(a, b) for a, b in zip(dists["ag1"], dists["ag2"])]
        G.add_edges_from(edge_list)
        nx.set_edge_attributes(
            G, dict(zip(edge_list, dists["Frobenius_distance"])), "frobDist"
        )

        dist_matrix = dists.pivot_table(
            index="ag1", columns="ag2", values="Frobenius_distance"
        ).fillna(0)
        dist_matrix = dist_matrix.reindex(
            index=G.nodes(), columns=G.nodes(), fill_value=0
        ).values

        dist_matrix = dist_matrix + dist_matrix.T - np.diag(np.diag(dist_matrix))
        mds = MDS(
            n_components=2, dissimilarity="precomputed", random_state=42, n_init=4
        )
        pos_array = mds.fit_transform(dist_matrix)

        pos = {node: pos_array[i] for i, node in enumerate(G.nodes())}
        if s == 18 and initial_w == 0.2:
            pos18_02_mds = pos
        nodes_o = [ag for ag in neighbours_dict.keys() if ag < 50]
        colors_o = [
            (simOut.loc[(simOut.time == t) & (simOut.agent_id == i), focal_att].iloc[0])
            for i in nodes_o
        ]
        nodes_s = [ag for ag in neighbours_dict.keys() if ag >= 50]
        colors_s = [
            (simOut.loc[(simOut.time == t) & (simOut.agent_id == i), focal_att].iloc[0])
            for i in nodes_s
        ]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes_s,
            node_color=colors_s,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            node_size=3,
            node_shape="<",
            ax=ax,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes_o,
            node_color=colors_o,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            node_size=3,
            node_shape="o",
            ax=ax,
        )
        mod = resNodes_df.loc[
            (resNodes_df.initial_w == params["initial_w"])
            & (resNodes_df.epsV == params["epsV"])
            & (resNodes_df.mu == params["mu"])
            & (resNodes_df.seed == params["seed"]),
            "modularity",
        ].values[0]
        sm = list(
            plt.cm.ScalarMappable(
                cmap="YlOrRd", norm=plt.Normalize(vmin=0, vmax=0.3)
            ).get_cmap()(mod)
        )
        sm[-1] = 0.4
        ax.set_facecolor(sm)
        ax.set_aspect("equal")

    cax = fig.add_axes([0.25, 0.03, 0.5, 0.02])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    plt.colorbar(
        sm, cax=cax, orientation="horizontal", ticks=[-1, 0, 1], label="focal belief"
    )

    plt.subplots_adjust(left=0.02, right=0.99, top=0.87, bottom=0.07)
    fig.text(
        0.98,
        -0.02,
        "background color =\n modularity of\n focal beliefs",
        fontsize=7,
        ha="right",
        va="bottom",
        transform=fig.transFigure,
    )
    fig.suptitle(
        "MDS of agent belief networks (Frob distance)\n"
        + rf"$\epsilon={epsV}$, $\mu={mu}$, $\omega_0 = {initial_w}$, $\rho={params['social_edge_weight']}$, $t={t}$"
    )
    plt.savefig(
        resultsfolder
        + "figs/"
        + f"mds_frobDists_epsV{epsV}_mu{mu}_initW{initial_w}_t{t}_rho{params['social_edge_weight']}{generate_filename(params)[1]}.pdf",
        bbox_inches="tight",
    )
# %%


def plot_BN(simOut, i, t, scaleE=3):
    ag = simOut.loc[(simOut.time == t) & (simOut.agent_id == i)]
    G = nx.Graph()
    for a in atts:
        G.add_node(a, value=ag[a])
    edgelist = list(combinations(atts, 2))
    for e in edgelist:
        G.add_edge(e[0], e[1], value=ag[f"({e[0]},{e[1]})"].values[0])

    widths = [scaleE * G.edges[e]["value"] for e in edgelist]
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(
        vmin=-1, vmax=1
    )  # 2. Convert scalar edge values to RGBA tuples
    edge_colors = [cmap(norm(G.edges[e]["value"])) for e in edgelist]

    fig = plt.figure(figsize=(8 / 2.54, 8 / 2.54))
    ax = plt.axes()
    pos = nx.circular_layout(G)
    nx.draw_networkx_edges(
        G=G, pos=pos, edge_color=edge_colors, width=widths, edgelist=edgelist
    )
    node_colors = [cmap(norm(ag[att])) for att in atts]
    nx.draw_networkx_nodes(
        G=G, pos=pos, nodelist=atts, node_color=node_colors, node_size=100
    )
    nx.draw_networkx_labels(G=G, pos=pos, labels=dict(zip(atts, atts)), font_size=7)
    ax.plot(
        pos[focal_att][0],
        pos[focal_att][1],
        marker="s",
        ms=15,
        markerfacecolor="None",
        markeredgecolor="grey",
        markeredgewidth=2,
        zorder=-1,
    )
    ax.axis(False)
    ax.set_facecolor((0, 0, 0, 0))
    return fig, ax


# %%
a = all_frob_dists["initW=0.2_epsV0.3_mu0.5_seed18"]
print(a.iloc[a["Frobenius_distance"].argmax()][["ag1", "ag2"]])

initial_w = 0.2
epsV = 0.3
mu = 0.5
s = 18
params.update(
    {
        "initial_w": initial_w,
        "epsV": epsV,
        "mu": mu,
        "seed": s,
    }
)
filename = generate_filename(params)[0]
simOut = pd.read_csv(filename + ".csv", low_memory=False)

for i in a.iloc[a["Frobenius_distance"].argmax()][["ag1", "ag2"]].values:
    fig, ax = plot_BN(simOut, i, 200)
    fig.text(
        0.9,
        0.84,
        f"Agent {i}",
        ha="right",
        va="top",
        transform=fig.transFigure,
        rotation=-45,
    )

# %%
plt.figure()
x = [p[0] for n, p in pos18_02_mds.items()]
y = [p[1] for n, p in pos18_02_mds.items()]
plt.scatter(x, y)
for i in a.iloc[a["Frobenius_distance"].argmax()][["ag1", "ag2"]].values:
    x = [p[0] for n, p in pos18_02_mds.items() if n in [i]]
    y = [p[1] for n, p in pos18_02_mds.items() if n in [i]]
    plt.scatter(x, y)
plt.gca().set_aspect("equal")
# %%
