# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import string
from netgraph import Graph
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
import seaborn as sns


plt.rcParams.update({"font.size": 10})

epsV = 0.3
mu = 0.5
lam = 0.0
s = 13
mem = 3
parties = ["A", "B"]
withinClusterP = 0.4
betweenClusterP = 0.01
socNetName = f"(Stoch-{len(parties)}-Block-{withinClusterP}-{betweenClusterP})"
n = 100
init_w = 0.2
M = 10

resultsfolder = "2025-07_results-dynNoB_velo/"

filename = (
    resultsfolder
    + f"dynamicNoB-_M-{M}_n-{n}-"
    + socNetName
    # + f"_beta-p{beta_pers}-s{beta_soc}"
    + f"_epsV{epsV}-m{mem}_eps{0.0}_mu{mu}_lam{lam}_initialW-{init_w}_seed{s}"
)

snapshots = pd.read_csv(filename + ".csv")

# %%
atts = list(string.ascii_lowercase[:M])
focal_att = "a"


def plot_BN(i, t, scaleE=3):
    ag = snapshots.loc[(snapshots.time == t) & (snapshots.agent_id == i)]
    G = nx.Graph()
    for a in atts:
        G.add_node(a, value=ag[a])
    edgelist = list(combinations(atts, 2))
    for e in edgelist:
        G.add_edge(e[0], e[1], value=ag[f"({e[0]},{e[1]})"].values[0])

    widths = [scaleE * G.edges[e]["value"] for e in edgelist]
    cmap = cm.get_cmap("coolwarm")
    norm = mcolors.Normalize(
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
    # ax.text(0, 0, rf"$t={t}$", ha="left", va="bottom", transform=ax.transAxes)
    # ax.text(
    #     1,
    #     0,
    #     f"agent {i}\ngroup {ag.identity.values[0]}",
    #     ha="right",
    #     va="bottom",
    #     transform=ax.transAxes,
    # )
    ax.axis(False)
    ax.set_facecolor((0, 0, 0, 0))
    return fig, ax


# %%
for i in [95, 93]:
    fig, ax = plot_BN(i, 100)
    ax.set_facecolor("pink")
    fig.savefig(f"bn{i}.png", dpi=300)
# %%

#################################
#####  Individual Level Personal Belief Network   #####
#################################


# %%

T = 100
edgeNames = [f"({a},{b})" for a, b in list(combinations(atts, 2))]
# Filter the snapshot at time T
from scipy.spatial.distance import pdist, squareform

# Filter snapshot at time T
df = snapshots.loc[snapshots.time == T, edgeNames + ["agent_id", "identity"]]
# Compute pairwise Euclidean distances
dist_array = pdist(df[edgeNames], metric="euclidean")
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

dists.iloc[dists["Frobenius_distance"].argmax()]
# %%

# Set up the figure and axes
fig, (ax_box, ax_kde) = plt.subplots(
    2,
    1,
    figsize=(16 / 2.54, 9 / 2.54),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 4], "hspace": 0.05},
)

# Boxplots (on top)
sns.boxplot(
    data=dists,
    x="Frobenius_distance",
    hue="groups",
    ax=ax_box,
    dodge=True,
    linewidth=1,
    fliersize=2,
)
ax_box.set(xlabel="", ylabel="")  # Remove labels for clarity
ax_box.legend_.remove()  # Remove legend to avoid duplication
ax_box.set_yticks([])

# KDE plot + histogram (on bottom)
sns.histplot(
    data=dists,
    x="Frobenius_distance",
    hue="groups",
    ax=ax_kde,
    alpha=0.2,
    kde=True,
    # element="step",
    stat="count",
    #    common_norm=False
)
ax_kde.set_ylim(0, min(ax_kde.get_ylim()[1], n * (n - 1) / 2 / 2))
# Optional: combine legends
ax_kde.legend(title="groups")
ax_box.set_title(rf"$\omega_0={init_w}$, $\epsilon_V={epsV},$ $\mu={mu}$, seed: {s}")

# Clean up visuals
# ax_box.set_xticks([])
sns.despine(ax=ax_box, bottom=True)
sns.despine(ax=ax_kde)
# %%
df = snapshots.loc[(snapshots.time == T), edgeNames + ["agent_id", "identity"]]
stdEdges = df.groupby("identity")[edgeNames].std()
stdEdges.loc["all", edgeNames] = df[edgeNames].std(axis=0)
stdEdges.T.sort_values("all").plot.bar()


# %%
import json


def plot_net(simOut, t, pos=None):
    neighbours_dict = {
        ag: json.loads(
            simOut.loc[
                (simOut.time == 0) & (simOut.agent_id == ag), "neighbours"
            ].values[0]
        )
        for ag in range(len(simOut.agent_id.unique()))
    }
    G = nx.from_dict_of_lists(neighbours_dict)
    fig = plt.figure()
    ax = plt.axes()
    colors = [
        (simOut.loc[(simOut.time == t) & (simOut.agent_id == i), focal_att])
        for i in neighbours_dict.keys()
    ]
    pos = nx.spring_layout(G, seed=1) if t > 0 else pos
    nx.draw_networkx_edges(G, pos=pos, width=0.5, alpha=0.5)
    nx.draw_networkx_nodes(
        G, pos=pos, node_color=colors, cmap="coolwarm", vmax=1, vmin=-1, node_size=100
    )
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1))
    sm._A = []
    plt.colorbar(sm, label="focal attitude", ticks=[-1, 0, 1], ax=ax)
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
    ax.set_alpha(0)
    return ax, pos


# %%
ax, pos = plot_net(
    snapshots,
    t=100,
)
ax.set_title(
    rf"$\beta_p = {beta_pers}, \beta_s={beta_soc}, \epsilon={epsV}, \mu={mu}$, seed={s}"
)

# %%
