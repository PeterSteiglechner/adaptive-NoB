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
import os

plt.rcParams.update({"font.size": 10})

# %%
epsV_on = 1.0
mu_on = 0.0
lam_on = 0.005
rho = 1.0
initial_ws = [0.2]
initial_w = initial_ws[0]
# %%

resultsfolder = "2025-11-02/"

T = 200
params = {
    "n": 100,
    "belief_options": np.linspace(-1, 1, 21),
    "social_edge_weight": rho,
    "memory": 1,
    "M": 10,
    "starBN": False,
    "depolarisation": False,
    "focal_att": "a",
    "initial_w": None,
    "beta_pers": 3.0,
    "epsV": None,
    "mu": None,
    "lam": None,
    "link_prob": 0.1,
    "seed": None,
    "T": T,
    "dt": 1,
    "track_times": np.arange(0, T + 1, 1),
    # "intervention_period": [],
    # "intervention_att": None,  # None,  # "b",
    # "intervention_strength": None,  # None,  # 10,  # 10,
    # "intervention_val": None,  # None,  # 1,  # 1,
}
params["atts"] = list(string.ascii_lowercase[: params["M"]])
params["edge_list"] = list(combinations(params["atts"], 2))
params["edgeNames"] = [f"({i},{j})" for i, j in params["edge_list"]]
atts = params["atts"]
edge_list = params["edge_list"]
edge_labels = [f"({i},{j})" for i, j in edge_list]
focal_att = params["focal_att"]

intervention_params = [
    "intervention_period",
    "intervention_att",
    "intervention_strength",
    "intervention_val",
]
noI = ([], None, None, None, "no intervention")
weak_focal = (range(100, 150), "a", 1, 1, "weak intervention on focal")
weak_nonfocal = (range(100, 150), "b", 1, 1, "weak intervention on non-focal")
medium_focal = (range(100, 150), "a", 2, 1, "strong intervention on focal")
strong_focal = (range(100, 150), "a", 4, 1, "strong intervention on focal")
xxstrong_focal = (range(100, 150), "a", 8, 1, "strong intervention on focal")
strong_nonfocal = (range(100, 150), "b", 4, 1, "strong intervention on non-focal")


def generate_filename(params):
    """Generate filename for results."""
    social_net = f"(p={params['link_prob']})"
    intervention = (
        f"_noIntervention"
        if params["intervention_att"] is None
        else f"_interv{params['intervention_period'][0]}-{params['intervention_period'][-1]}-{params['intervention_att']}-strength{params['intervention_strength']}-value{params['intervention_val']}"
    )
    return (
        f"{resultsfolder}sims/adaptiveBN_M-{params['M']}{'star' if params['starBN'] else ''}-{'depolInitial' if params["depolarisation"] else 'randInitial'}_n-{params['n']}-{social_net}"
        f"_epsV{params['epsV']}-m{params['memory']}_mu{params['mu']}"
        f"_lam{params['lam']}_rho{params['social_edge_weight']}_beta{params['beta_pers']}_initialW-{params['initial_w']}"
        f"{intervention}"
        f"_seed{params['seed']}"
    ), intervention


seeds = range(30, 60)
interventions = [
    "noI",
    "weak_focal",
    "medium_focal",
    "strong_focal",
    "xxstrong_focal",
    # "weak_nonfocal",
    # "strong_nonfocal",
]
selected_seeds = seeds
print(selected_seeds)
paramCombis = [
    (initial_w, epsV, mu, lam, s, intervention_name)
    for initial_w in initial_ws  # [0.05, 0.8]
    for epsV, mu, lam in [(0.0, 0.0, 0.0), (epsV_on, mu_on, lam_on)]
    for s in seeds
    for intervention_name in interventions
]


# %%#################################
#####  GIF of BNs   #####
#################################
def plot_BN(simOut, i, t, params, intervention_name, scaleE=1, scaleN=100):
    ag = simOut.loc[(simOut.time == t) & (simOut.agent_id == i)]
    G = nx.Graph()
    for a in atts:
        G.add_node(a, value=ag[a])
    edgelist = list(combinations(atts, 2))
    for e in edgelist:
        G.add_edge(e[0], e[1], value=ag[f"({e[0]},{e[1]})"].values[0])
    # print(ag.loc[:, edge_labels])

    widths = [scaleE * G.edges[e]["value"] for e in edgelist]
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(
        vmin=-1, vmax=1
    )  # 2. Convert scalar edge values to RGBA tuples
    edge_colors = [cmap(norm(G.edges[e]["value"])) for e in edgelist]

    fig = plt.figure(figsize=(8 / 2.54, 8 / 2.54))
    ax = plt.axes()
    # pos = nx.circular_layout(G)
    pos = nx.spring_layout(G, weight="value", seed=21, k=2)
    nx.draw_networkx_edges(
        G=G, pos=pos, edge_color=edge_colors, width=widths, edgelist=edgelist
    )
    node_colors = [cmap(norm(ag[att])) for att in atts]
    nx.draw_networkx_nodes(
        G=G, pos=pos, nodelist=atts, node_color=node_colors, node_size=scaleN
    )
    nx.draw_networkx_labels(G=G, pos=pos, labels=dict(zip(atts, atts)), font_size=7)
    ax.plot(
        pos[focal_att][0],
        pos[focal_att][1],
        marker="s",
        ms=20,
        markerfacecolor="None",
        markeredgecolor="grey",
        markeredgewidth=4,
        zorder=-1,
    )
    ax.text(
        0.02,
        0.02,
        f"$t={t:3d}$" + "\n" + f"ag {i}",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )
    if not intervention_name == "noI":
        ax.text(
            0.98,
            0.98,
            (
                f"{('after ' if t>params["intervention_period"][-1] else '')} {eval(intervention_name)[-1]}"
                if t > params["intervention_period"][0]
                else ("")
            ),
            ha="right",
            va="bottom",
            transform=ax.transAxes,
        )
    # ax.text(
    #     1,
    #     0.98,
    #     rf"$\epsilon={params['epsV']}$"
    #     + "\n"
    #     + rf"$\mu={params['mu']}$,"
    #     + "\n"
    #     + rf"$\lambda={params['lam']}$",
    #     ha="right",
    #     va="top",
    #     transform=ax.transAxes,
    #     fontsize=8,
    # )

    ax.axis(False)
    ax.set_facecolor((0, 0, 0, 0))
    return fig, ax


# %%
#################################
#################################
#################################
#################################
#################################
#####  PARAMS    #####
#################################
#################################
#################################
#################################
#################################

intervention_name = "medium_focal"
seed = 42
params.update(dict(zip(intervention_params, eval(intervention_name))))
params.update(
    {
        "initial_w": initial_w,
        "epsV": epsV_on,
        "mu": mu_on,
        "lam": lam_on,
    }
)
params["seed"] = seed
filename = generate_filename(params)[0]
simOut = pd.read_csv(filename + ".csv", low_memory=False)
avg = 10
times_before = np.arange(100 - avg + 1, 101)
times_short_after = np.arange(150 - avg + 1, 151)
times_long_after = np.arange(200 - avg + 1, 201)
ops_before = (
    simOut.loc[simOut.time.isin(times_before)].groupby("agent_id")[focal_att].mean()
)
ops_short_after = (
    simOut.loc[simOut.time.isin(times_short_after)]
    .groupby("agent_id")[focal_att]
    .mean()
)
ops_long_after = (
    simOut.loc[simOut.time.isin(times_long_after)].groupby("agent_id")[focal_att].mean()
)
prior_negs = ops_before[ops_before < 0].index
if len(prior_negs) > 0:
    # Agents who flipped to positive short-term
    temporary_flippers = ops_short_after.loc[prior_negs][ops_short_after > 0].index
    longterm_flippers = ops_long_after.loc[prior_negs][ops_long_after > 0].index
reverters = ops_long_after.loc[temporary_flippers][
    ops_long_after.loc[temporary_flippers] < 0
].index
resistant = ops_long_after.loc[prior_negs][ops_long_after.loc[prior_negs] < 0].index

print(len(prior_negs))
display(reverters), display(resistant), display(longterm_flippers)

# %%
gif_folder = resultsfolder + "_gif/"
times = np.arange(0, 201, 1)  # [::-1]
for i in [18, 66, 3]:  # a.loc[a["a"] < 0, "agent_id"]:
    if not os.path.isdir(gif_folder + f"{i}/"):
        os.mkdir(gif_folder + f"{i}/")
    for t in times:
        fig, ax = plot_BN(simOut, i, t, params, intervention_name, scaleN=200)

        plt.savefig(gif_folder + f"{i}/t{t}_{intervention_name}_spring.png")
        plt.close()
# %%
from PIL import Image

images = []
i = 3
if True:  # len(times) == 200:
    for t in np.arange(0, 201, 10):
        im = Image.open(gif_folder + f"{i}/t{t}_{intervention_name}_spring.png")
        images.append(im)
    last_frame = len(images)
    for x in range(0, 9):
        im = images[last_frame - 1]
        images.append(im)
    images[0].save(
        resultsfolder
        + "figs/"
        + f"snaps_{intervention_name}-{initial_w}_{epsV_on}-{mu_on}_seed{seed}_ag{i}_t0-200_spring.gif",
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=1000,
        loop=0,
    )

    print(
        resultsfolder
        + "figs/"
        + f"snaps_{intervention_name}-{initial_w}_{epsV_on}-{mu_on}_seed{seed}_ag{i}_t0-200.gif",
    )

# %%

#################################
#####  PLOT TIME SERIES   #####
#################################

simRes = simOut.loc[simOut.time == 200]
df = simRes[edge_labels + ["agent_id", "identity"]]

grouped = (
    simOut.groupby(["time", "identity"])[atts + edge_labels]
    .agg(["mean", "std"])
    .reset_index()
)
fig, axes = plt.subplot_mosaic(
    [["ax", "ax_kdeA", "ax_kdeB"], ["ax2", "ax2_kdeA", "ax2_kdeB"]],
    figsize=(12, 8),
    width_ratios=[3, 0.5, 0.5],
)

ax, ax_kdeA, ax_kdeB = axes["ax"], axes["ax_kdeA"], axes["ax_kdeB"]
ax2, ax2_kdeA, ax2_kdeB = axes["ax2"], axes["ax2_kdeA"], axes["ax2_kdeB"]

epsV = epsV_on
t_focus = 200
# ============= PLOT ATTRIBUTES (left panel) =============
nBN = 0
for ni, identity in enumerate(grouped["identity"].unique()):
    sub = grouped[grouped["identity"] == identity]
    color = plt.get_cmap("Set1")(ni + nBN * 2)

    for att in atts:
        if not att == focal_att:
            ax.plot(
                sub["time"],
                sub[(att, "mean")],
                color=color,
                alpha=0.2,
                lw=0.4,
                ls="--" if epsV == 0 else "-",
                label=f"_",  # {'adaptive BN' if epsV>0 else 'fixed BN'} {identity}",
            )
            ax.fill_between(
                sub["time"],
                sub[(att, "mean")] - sub[(att, "std")],
                sub[(att, "mean")] + sub[(att, "std")],
                alpha=0.02,
                color=color,
            )

    ax.plot(
        sub["time"],
        sub[(focal_att, "mean")],
        color=color,
        ls="--" if epsV == 0 else "-",
        label=f"{'adaptive BN' if epsV>0 else 'fixed BN'} {identity}",
    )
    ax.fill_between(
        sub["time"],
        sub[(focal_att, "mean")] - sub[(focal_att, "std")],
        sub[(focal_att, "mean")] + sub[(focal_att, "std")],
        alpha=0.2,
        color=color,
    )
    ax.plot(
        [],
        [],
        ls="--" if epsV == 0 else "-",
        lw=0.4,
        alpha=0.2,
        color=color,
        label="other beliefs",
    )

    # --- KDE on right panel ---
    raw_vals = simOut.loc[
        (simOut["time"] == 99) & (simOut["identity"] == identity),
        focal_att,
    ]
    if len(raw_vals) > 1:
        sns.histplot(
            y=raw_vals,
            ax=ax_kdeA,
            fill=True,
            alpha=0.4 if epsV > 0 else 0.4,
            color=color,
            label=identity,
            bins=np.linspace(-1 - 1 / 14, 1 + 1 / 14, 14),
        )
    # --- KDE on right panel ---
    raw_vals = simOut.loc[
        (simOut["time"] == t_focus) & (simOut["identity"] == identity),
        focal_att,
    ]
    if len(raw_vals) > 1:
        sns.histplot(
            y=raw_vals,
            ax=ax_kdeB,
            fill=True,
            alpha=0.4 if epsV > 0 else 0.4,
            color=color,
            label=identity,
            bins=np.linspace(-1 - 1 / 14, 1 + 1 / 14, 14),
        )

# ============= PLOT EDGES (left panel) =============
edge_labels_sorted = [e for e in edge_labels if focal_att in e] + [
    e for e in edge_labels if focal_att not in e
]
for ne, e in enumerate(edge_labels_sorted):
    for identity in grouped["identity"].unique():
        sub = grouped[grouped["identity"] == identity]
        color = plt.get_cmap("Set1")(ne)  # (2 * ne + (identity == "B"))

        ax2.plot(
            sub["time"],
            sub[(e, "mean")],
            label=f"{identity}: {e}" if epsV > 0 and focal_att in e else "_",
            color=color if focal_att in e else "#008080",
            ls=":" if identity == "A" else "-.",
            alpha=0.2 if epsV == 0 or not focal_att in e else 0.6,
            lw=0.5 if epsV == 0 or not focal_att in e else 3,
            zorder=10 if focal_att in e else 8,
        )
        ax2.fill_between(
            sub["time"],
            sub[(e, "mean")] - sub[(e, "std")],
            sub[(e, "mean")] + sub[(e, "std")],
            label="_",
            color=color if focal_att in e else "#008080",
            # ls="--" if identity == "A" else "-",
            alpha=0.05 if epsV == 0 or not focal_att in e else 0.1,
            zorder=7 if focal_att in e else 4,
        )
        if epsV > 0:
            # --- KDE on right panel ---
            raw_vals = simOut.loc[
                (simOut["time"] == 99) & (simOut["identity"] == identity),
                e,
            ]
            sns.kdeplot(
                y=raw_vals,
                ax=ax2_kdeA,
                fill=False,
                alpha=0.4 if epsV > 0 else 0.2,
                color=color if focal_att in e else "#008080",
                label=f"{identity}: {e}",
                zorder=-1,
            )
            ax2.set_ylim(-1, 3)
            # --- KDE on right panel ---
            raw_vals = simOut.loc[
                (simOut["time"] == t_focus) & (simOut["identity"] == identity),
                e,
            ]
            sns.kdeplot(
                y=raw_vals,
                ax=ax2_kdeA if identity == "A" else ax2_kdeB,
                fill=False,
                alpha=0.4 if epsV > 0 else 0.2,
                color=color if focal_att in e else "#008080",
                label=f"{identity}: {e}",
                zorder=-1,
            )
            ax2.set_ylim(-1, 3)

# === cosmetics ===
if not intervention_name == "noI":
    for ax_ in [ax, ax2]:
        ax_.fill_between(
            [100, 150],
            [ax_.get_ylim()[0], ax_.get_ylim()[0]],
            [ax_.get_ylim()[1], ax_.get_ylim()[1]],
            color="gainsboro",
            zorder=-1,
        )
    ax.text(
        125,
        ax.get_ylim()[1] + 0.1,
        f"{eval(intervention_name)[4]}",
        ha="center",
        va="bottom",
    )
ax.legend(fontsize=6, ncol=2)
ax2.legend(loc="upper left", fontsize=6, ncol=4)  # bbox_to_anchor=(1.0, 1.02)
for axK in [ax_kdeA, ax_kdeB]:
    axK.sharey(ax)
    axK.set_ylabel("")
    axK.set_xticks([])
    # axK.set_yticklabels([])
for axK in [ax2_kdeA, ax2_kdeB]:
    axK.sharey(ax2)
    axK.set_ylabel("")
    axK.set_xticks([])
    # axK.set_yticklabels([])
ax.set_ylabel("focal/non-focal belief (mean + sd per group)")
ax2.set_ylabel("edge weights (mean + sd per group)")
ax2.set_xlabel("time")
ax_kdeA.text(10, -1.5, "focal belief", fontsize=10)
ax_kdeB.text(10, -1.5, "focal belief", fontsize=10)
ax_kdeA.set_title("Distributions in groups $A$ and $B$\nat $t=200$", x=1)
fig.tight_layout()
plt.savefig(resultsfolder + "figs/" + f"{filename[len(resultsfolder):]}.png")

# %%
#####  DISTS   #####

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


ags = dists.iloc[dists.Frobenius_distance.argmax()][["ag1", "ag2"]]
print(ags)
# %%


a = simOut.loc[simOut.time == 200]
b = a.loc[a["a"] < 0, [e for e in edge_labels if "a" in e]]
a[[e for e in edge_labels if "a" in e]].plot.hist(alpha=0.3, bins=np.linspace(-1, 1.5))
cols = plt.get_cmap("Set1")
for n, (id, bb) in enumerate(b.iterrows()):
    plt.vlines(bb, 0, 25, color=cols(n / len(b)))

# %%

intervention_name = "strong_focal"

params.update(dict(zip(intervention_params, eval(intervention_name))))
params.update(
    {
        "initial_w": initial_w,
        "epsV": epsV_on,
        "mu": mu_on,
        "lam": lam_on,
    }
)
params["seed"] = 31
filename = generate_filename(params)[0]
simOut = pd.read_csv(filename + ".csv", low_memory=False)

plt.figure(figsize=(8, 3))
ax = plt.axes()
a = simOut.loc[simOut.time == 200]

b = a.loc[a["a"] < 0, [e for e in edge_labels if "a" in e]]
id = simOut.loc[b.index, "agent_id"]
for i in id:
    (simOut.loc[simOut.agent_id == i, :]).plot(
        x="time", y="a", lw=0.4, ax=ax, legend=False
    )

ax.set_ylabel("focal belief")
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_ylim(-1, 1)
ax.set_xlim(0, 200)
ax.fill_between([100, 150], [-1, -1], [1, 1], color="darkred", alpha=0.1)

plt.figure(figsize=(8, 3))
ax = plt.axes()
b = a.loc[a["a"] > 0, [e for e in edge_labels if "a" in e]]
id = simOut.loc[b.index, "agent_id"]
for i in id:
    (simOut.loc[simOut.agent_id == i, :]).plot(
        x="time", y="a", lw=0.4, color="grey", ax=ax, legend=False
    )

ax.set_ylabel("focal belief")
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_ylim(-1, 1)
ax.set_xlim(0, 200)
ax.fill_between([100, 150], [-1, -1], [1, 1], color="darkred", alpha=0.1)


# %%
# intervention_name = "medium_focal"

# params.update(dict(zip(intervention_params, eval(intervention_name))))
# params.update(
#     {
#         "initial_w": initial_w,
#         "epsV": epsV_on,
#         "mu": mu_on,
#         "lam": lam_on,
#     }
# )
# params["seed"] = 42
# filename = generate_filename(params)[0]


# fig, axs = plt.subplots(1, 2, figsize=(8, 3))
# arr = []
# selected_seeds = seeds
# for s in selected_seeds:
#     params["seed"] = s
#     filename = generate_filename(params)[0]
#     simOut = pd.read_csv(filename + ".csv", low_memory=False)
#     conformists = simOut.loc[
#         (simOut.time == 200) & (simOut[focal_att] > 0), "agent_id"
#     ].values
#     nonconformists = simOut.loc[
#         (simOut.time == 200) & (simOut[focal_att] < 0), "agent_id"
#     ].values
#     robusts = simOut.loc[
#         (simOut.time == 150)
#         & (simOut[focal_att] < 0)
#         & (simOut.agent_id.isin(nonconformists)),
#         "agent_id",
#     ].values
#     resiliant = simOut.loc[
#         (simOut.time == 150)
#         & (simOut[focal_att] > 0)
#         & (simOut.agent_id.isin(nonconformists)),
#         "agent_id",
#     ].values
#     a = simOut.loc[simOut.time == 200]
#     a["conformist"] = a.apply(lambda x: x.agent_id in conformists, axis=1)
#     a["robust"] = a.apply(
#         lambda x: x.agent_id in nonconformists and x.agent_id in robusts, axis=1
#     )
#     a
#     arr.append(a)
# a = pd.concat(arr)
# # b = a.loc[a["a"] < 0, [e for e in edge_labels if "a" in e]]
# # id = simOut.loc[b.index, "agent_id"]
# # b.abs().mean()
# edges_focal = [e for e in edge_labels if "a" in e]
# edges_noncal = [e for e in edge_labels if "a" not in e]
# for n, edges in enumerate([edges_focal, edges_noncal]):
#     for who, color in [
#         (a, "lightgrey"),
#         (a.loc[a.conformist], "red"),
#         (a.loc[(~a.conformist) & (a.robust)], "blue"),
#         (a.loc[(~a.conformist) & (~a.robust)], "lime"),
#     ]:
#         axs[n].hist(
#             who[edges].abs().mean(axis=1),
#             bins=np.linspace(0, 1.5, 60),
#             color=color,
#             alpha=0.2,
#         )
# axs[1].set_yticks([])
# axs[0].set_yticks([])

# %%

# %%


# fig, axs = plt.subplots(1, 2, figsize=(8, 3))
# arr99 = []
# selected_seeds = [31]
# for s in selected_seeds:
#     params["seed"] = s
#     filename = generate_filename(params)[0]
#     simOut = pd.read_csv(filename + ".csv", low_memory=False)
#     a99 = simOut.loc[simOut.time == 200]
#     arr99.append(a99)
# a99 = pd.concat(arr99)

# resilients = a99.loc[a99.agent_id.isin(a.loc[a["a"] < 0].agent_id)]
# conformists = a99.loc[a99.agent_id.isin(a.loc[a["a"] > 0].agent_id)]

# resilientsEdgesFocal = np.histogram(
#     resilients[[e for e in edge_labels if "a" in e]].abs().mean(axis=1),
#     bins=np.linspace(0, 3, 60),
# )
# conformistsEdgesFocal = np.histogram(
#     conformists[[e for e in edge_labels if "a" in e]].abs().mean(axis=1),
#     bins=np.linspace(0, 3, 60),
# )
# axs[0].bar(
#     resilientsEdgesFocal[1][:-1] + np.diff(resilientsEdgesFocal[1])[0] / 2,
#     resilientsEdgesFocal[0] / conformistsEdgesFocal[0],
#     label="focal – non-focal",
# )  # /

# resilientsEdgesNonFocal = np.histogram(
#     resilients[[e for e in edge_labels if "a" not in e]].abs().mean(axis=1),
#     bins=np.linspace(0, 3, 60),
# )
# conformistsEdgesNonFocal = np.histogram(
#     conformists[[e for e in edge_labels if "a" not in e]].abs().mean(axis=1),
#     bins=np.linspace(0, 3, 60),
# )
# axs[1].bar(
#     resilientsEdgesNonFocal[1][:-1] + np.diff(resilientsEdgesNonFocal[1])[0] / 2,
#     resilientsEdgesNonFocal[0] / conformistsEdgesNonFocal[0],
#     label="non-focal",
# )  # /
# plt.legend()


# plt.ylabel("fraction of resilient over conformists")
# plt.xlabel("edge weight")
# %%


fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
for n, id in enumerate(a.loc[a["a"] > 0, "agent_id"].sample(5)):
    c = [
        plt.color_sequences["tab10"][k] if "a" in e else "grey"
        for k, e in enumerate(edge_labels_sorted)
    ][::-1]
    simOut.loc[simOut.agent_id == id, ["time"] + edge_labels_sorted[::-1]].plot(
        x="time", color=c, legend=False, ax=axs[0, n]
    )
    ax.set_ylim(-4, 4)
for n, id in enumerate(a.loc[a["a"] < 0, "agent_id"].sample(5)):
    c = [
        plt.color_sequences["tab10"][k] if "a" in e else "grey"
        for k, e in enumerate(edge_labels_sorted)
    ][::-1]
    simOut.loc[simOut.agent_id == id, ["time"] + edge_labels_sorted[::-1]].plot(
        x="time", color=c, legend=False, ax=axs[1, n]
    )
    ax.set_ylim(-4, 4)

# %%


# %%
def plot_BN_ax(simOut, i, t, ax, params, intervention_name, scaleE=1, scaleN=100):
    ag = simOut.loc[(simOut.time == t) & (simOut.agent_id == i)]
    G = nx.Graph()
    for a in atts:
        G.add_node(a, value=ag[a])
    edgelist = list(combinations(atts, 2))
    for e in edgelist:
        G.add_edge(e[0], e[1], value=ag[f"({e[0]},{e[1]})"].values[0])
    widths = [scaleE * G.edges[e]["value"] for e in edgelist]
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-1, vmax=1)
    edge_colors = [cmap(norm(G.edges[e]["value"])) for e in edgelist]
    pos = nx.spring_layout(G, weight="value", seed=4242, k=2)
    nx.draw_networkx_edges(
        G=G, pos=pos, edge_color=edge_colors, width=widths, edgelist=edgelist, ax=ax
    )
    node_colors = [cmap(norm(ag[att])) for att in atts]
    nx.draw_networkx_nodes(
        G=G, pos=pos, nodelist=atts, node_color=node_colors, node_size=scaleN, ax=ax
    )
    nx.draw_networkx_labels(
        G=G, pos=pos, labels=dict(zip(atts, atts)), font_size=7, ax=ax
    )
    ax.plot(
        pos[focal_att][0],
        pos[focal_att][1],
        marker="s",
        ms=15,
        markerfacecolor="None",
        markeredgecolor="orange",
        markeredgewidth=3,
        zorder=-1,
    )
    if not intervention_name == "noI":
        if (
            t > params["intervention_period"][0]
            and t <= params["intervention_period"][-1]
        ):
            ax.text(
                0.5,
                1,
                f"intervention",
                va="top",
                ha="center",
                transform=ax.transAxes,
                color="orange",
            )
    ax.axis(False)
    ax.set_facecolor((0, 0, 0, 0))


def plot_dynamics(simOut, i, t, ax, params, intervention_name):
    for ag in simOut.agent_id.unique():
        simOut.loc[simOut.agent_id == ag].plot(
            ax=ax,
            x="time",
            y=focal_att,
            color="grey",
            lw=0.5,
            alpha=0.5,
            legend=False,
            label="_",
        )
    print(simOut.loc[(simOut.time == t) & (simOut[focal_att] < 0), "agent_id"].count())
    simOut.loc[(simOut.agent_id == i) & (simOut.time <= t)].plot(
        ax=ax,
        x="time",
        y=focal_att,
        color="red",
        lw=3,
        alpha=0.5,
        legend=False,
        label="_",
    )
    simOut.loc[(simOut.agent_id == i) & (simOut.time == t)].plot(
        ax=ax,
        x="time",
        y=focal_att,
        color="red",
        marker="o",
        ms=6,
        alpha=0.5,
        legend=False,
        label="_",
    )
    if not intervention_name == "noI":
        ax.fill_between(
            params["intervention_period"],
            [-1] * len(params["intervention_period"]),
            [1] * len(params["intervention_period"]),
            color="red",
            alpha=0.05 * params["intervention_val"],
            zorder=-1,
        )
    ax.set_yticks([-1, 0, 1])
    ax.set_ylabel("focal belief")
    ax.set_ylim(-1.01, 1.01)
    ax.set_xlim(0, 200)
    ax.set_yticks([-1, 0, 1])
    ax.set_xticks([0, 100, 200])
    ax.plot([], [], lw=1, color="grey", alpha=0.5, label="single agent")
    ax.legend(loc="upper left")
    return


# %%
intervention_name = "medium_focal"
seed = 42
params.update(dict(zip(intervention_params, eval(intervention_name))))
params.update(
    {
        "initial_w": initial_w,
        "epsV": epsV_on,
        "mu": mu_on,
        "lam": lam_on,
    }
)
params["seed"] = seed
filename = generate_filename(params)[0]
simOut = pd.read_csv(filename + ".csv", low_memory=False)
avg = 10
times_before = np.arange(100 - avg + 1, 101)
times_short_after = np.arange(150 - avg + 1, 151)
times_long_after = np.arange(200 - avg + 1, 201)
ops_before = (
    simOut.loc[simOut.time.isin(times_before)].groupby("agent_id")[focal_att].mean()
)
ops_short_after = (
    simOut.loc[simOut.time.isin(times_short_after)]
    .groupby("agent_id")[focal_att]
    .mean()
)
ops_long_after = (
    simOut.loc[simOut.time.isin(times_long_after)].groupby("agent_id")[focal_att].mean()
)
prior_negs = ops_before[ops_before < 0].index
if len(prior_negs) > 0:
    # Agents who flipped to positive short-term
    temporary_flippers = ops_short_after.loc[prior_negs][ops_short_after > 0].index
    longterm_flippers = ops_long_after.loc[prior_negs][ops_long_after > 0].index
reverters = ops_long_after.loc[temporary_flippers][
    ops_long_after.loc[temporary_flippers] < 0
].index
resistant = ops_long_after.loc[prior_negs][ops_long_after.loc[prior_negs] < 0].index

print(len(prior_negs))
display(reverters), display(resistant), display(longterm_flippers)

# %%

gif_folder = resultsfolder + "_gif/"
times = np.arange(0, 201, 1)  # [::-1]
for i in [18, 66, 3]:  # a.loc[a["a"] < 0, "agent_id"]:
    if not os.path.isdir(gif_folder + f"{i}/"):
        os.mkdir(gif_folder + f"{i}/")
    for t in times:
        fig, axs = plt.subplots(
            1, 2, width_ratios=[3, 1], figsize=(16 / 2.54, 7 / 2.54)
        )
        plot_BN_ax(simOut, i, t, axs[1], params, intervention_name, scaleN=80)
        plot_dynamics(simOut, i, t, axs[0], params, intervention_name)
        # axs[0].fill_between(
        #     [100, 150], [-1, -1], [1, 1], zorder=-1, color="orange", alpha=0.1
        # )
        fig.tight_layout()
        plt.savefig(gif_folder + f"{i}/t{t}_{intervention_name}_spring.png")
        plt.close()
# %%

from PIL import Image

images = []
i = 66
if True:  # len(times) == 200:
    for t in np.arange(0, 201, 1):
        im = Image.open(gif_folder + f"{i}/t{t}_{intervention_name}_spring.png")
        images.append(im)
    last_frame = len(images)
    for x in range(0, 9):
        im = images[last_frame - 1]
        images.append(im)
    images[0].save(
        resultsfolder
        + "figs/"
        + f"snapsFull_{intervention_name}-{initial_w}_{epsV_on}-{mu_on}_seed{seed}_ag{i}_t0-200_spring.gif",
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=50,
        loop=0,
    )


# %%
def plot_dynamics_all(simOut, ids, t, ax, params, intervention_name):
    cols = dict(zip(ids, plt.color_sequences["Set1"]))
    for ag in simOut.agent_id.unique():
        if not ag in ids:
            simOut.loc[(simOut.agent_id == ag) & (simOut.time <= t)].plot(
                ax=ax,
                x="time",
                y=focal_att,
                color="grey",
                lw=0.5,
                alpha=0.5,
                legend=False,
                label="_",
                zorder=2,
                clip_on=False,
            )
        else:
            simOut.loc[(simOut.agent_id == ag) & (simOut.time <= t)].plot(
                ax=ax,
                x="time",
                y=focal_att,
                color=cols[ag],
                lw=1,
                alpha=0.5,
                legend=False,
                label="_",
                clip_on=False,
                zorder=5,
            )
            simOut.loc[(simOut.agent_id == ag) & (simOut.time == t)].plot(
                ax=ax,
                x="time",
                y=focal_att,
                color=cols[ag],
                marker="o",
                alpha=0.5,
                legend=False,
                label="_",
                clip_on=False,
                zorder=5,
            )
    if not intervention_name == "noI":
        ax.fill_between(
            params["intervention_period"],
            [-1] * len(params["intervention_period"]),
            [1] * len(params["intervention_period"]),
            color="red",
            alpha=0.2 * params["intervention_val"],
            zorder=-1,
        )
    ax.set_yticks([-1, 0, 1])
    ax.set_ylabel("focal belief")
    ax.set_ylim(-1.01, 1.01)
    ax.set_xlim(0, 200)
    ax.set_yticks([-1, 0, 1])
    ax.set_xticks([0, 100, 200])
    ax.plot([], [], lw=1, color="grey", alpha=0.5, label="single agent")
    ax.legend(loc="upper left")
    return


# %%
intervention_name = "medium_focal"
seed = 31
params.update(dict(zip(intervention_params, eval(intervention_name))))
params.update(
    {
        "initial_w": initial_w,
        "epsV": epsV_on,
        "mu": mu_on,
        "lam": lam_on,
    }
)
params["seed"] = seed
filename = generate_filename(params)[0]
simOut = pd.read_csv(filename + ".csv", low_memory=False)

gif_folder = resultsfolder + "_gif/"
times = np.arange(0, 201, 1)  # [::-1]
if not os.path.isdir(gif_folder + f"dyn/"):
    os.mkdir(gif_folder + f"dyn/")
ids = np.random.choice(simOut.agent_id.unique(), replace=False, size=8)
for t in times:
    if t in range(0, 200, 50):
        print(t, end=", ")
    fig, ax = plt.subplots(1, 1, figsize=(12 / 2.54, 7 / 2.54))
    plot_dynamics_all(simOut, ids, t, ax, params, intervention_name)
    fig.tight_layout()
    plt.savefig(
        gif_folder + f"dyn/t{t}_{intervention_name}_seed{seed}_eps{params['epsV']}.png"
    )
    plt.close()

# %%
images = []
for t in np.arange(0, 201, 1):
    im = Image.open(
        gif_folder + f"dyn/t{t}_{intervention_name}_seed{seed}_eps{params['epsV']}.png"
    )
    images.append(im)
last_frame = len(images)
for x in range(0, 50):
    im = images[last_frame - 1]
    images.append(im)
images[0].save(
    resultsfolder
    + "figs/"
    + f"snapsDyn_{intervention_name}-{initial_w}_{params['epsV']}_seed{seed}_t0-200.gif",
    save_all=True,
    append_images=images[1:],
    optimize=False,
    duration=30,
    loop=0,
)
# %%
