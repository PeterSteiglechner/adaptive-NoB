#################################
#####  Plot FOcal Nets at t=T   #####
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

plt.rcParams.update({"font.size": 10})
# plt.rcParams.update({"font.size":10})

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
    "beta_pers": 3.0,
    "memory": 1,
    "M": 10,
    "starBN": False,
    "depolarisation": False,
    "focal_att": "a",
    "initial_w": None,
    "epsV": None,
    "mu": None,
    "lam": None,
    # "clusters": ["A", "B"],
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
medium_focal = (range(100, 150), "a", 2, 1, "medium intervention on focal")
strong_focal = (range(100, 150), "a", 4, 1, "strong intervention on focal")
xxstrong_focal = (range(100, 150), "a", 8, 1, "strong intervention on focal")
weak_nonfocal = (range(100, 150), "b", 1, 1, "weak intervention on non-focal")
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


# %%

seeds = range(30, 60)

selected_seeds = seeds[
    ::-1
]  # np.random.choice(seeds, replace=False, size=min(20, len(seeds)))
print(selected_seeds)
paramCombis = [
    #  (init_w, epsV, mu, seed)
    (initial_w, epsV, mu, lam, s, intervention_name)
    for initial_w in initial_ws  # [0.05, 0.8]
    for epsV, mu, lam in [(epsV_on, mu_on, lam_on)]  # (0.0, 0.0, 0.0),
    for s in seeds
    for intervention_name in [
        "noI",
        "weak_focal",
        "medium_focal",
        "strong_focal",
        "xxstrong_focal",
        # "weak_nonfocal",
        # "strong_nonfocal",
    ]
]
# %%


def plot_net(
    simOut, t, params, mod="white", pos=None, ax=None, cbar=True, node_size=3, cax=None
):
    neighbours_dict = {
        ag: json.loads(
            simOut.loc[
                (simOut.time == 0) & (simOut.agent_id == ag), "neighbours"
            ].values[0]
        )
        for ag in range(0, params["n"])
    }
    G = nx.from_dict_of_lists(neighbours_dict)
    if t == 200:
        simRes = simOut.loc[(simOut.time == t)].sort_values("agent_id")
        # simRes["avg_final_focal"] = (
        #     simOut.loc[simOut.time.isin(list(range(190, 201)))]
        #     .groupby("agent_id")["a"]
        #     .mean()
        #     .sort_index()
        #     .values
        # )
        focal_att = params["focal_att"]
    else:
        simRes = simOut.loc[(simOut.time == t)].sort_values("agent_id")
        focal_att = params["focal_att"]

    colors = [
        (simRes.loc[(simOut.agent_id == i), focal_att].iloc[0])
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
        node_size=node_size,
        ax=ax,
    )
    if cbar:
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1))
        sm._A = []
        plt.colorbar(
            sm,
            label=("focal belief" + " (avg of 10 steps)" if focal_att == "a" else ""),
            ticks=[-1, 0, 1],
            # ax=ax,
            orientation="horizontal",
            pad=0.05,
            cax=cax,
        )
    if not mod == "white":
        sm = list(
            plt.cm.ScalarMappable(cmap="YlGn").get_cmap()(
                plt.Normalize(vmin=0, vmax=0.5)(mod)
            )
        )
        sm[-1] = 0.4
        ax.set_facecolor(sm)
    return ax, pos


def plot_nets_tot(
    t, epsV, mu, lam, initial_w, selected_seeds, intervention_name, sorted=False
):
    params.update({"initial_w": initial_w, "epsV": epsV, "mu": mu, "lam": lam})
    params.update(dict(zip(intervention_params, eval(intervention_name))))

    fig, axs = plt.subplots(1, min(7, len(selected_seeds)), figsize=(12, 3))
    # mod = res.loc[
    #     (res.initial_w == params["initial_w"])
    #     & (res.epsV == params["epsV"])
    #     & (res.mu == params["mu"])
    #     & (res.intervention_name == intervention_name)
    #     & (res.seed.isin(selected_seeds)),
    # ]
    # mod = mod.set_index("seed").loc[selected_seeds].reset_index()

    for s, ax in zip(selected_seeds, axs.flatten()):
        params["seed"] = s
        filename = generate_filename(params)[0]
        simOut = pd.read_csv(filename + ".csv", low_memory=False)

        plot_net(
            simOut,
            t,
            params,
            # mod.loc[mod.seed == s, "modularity"].values[0],
            ax=ax,
            cbar=False,  # (ax == axs[-2]),
            # cax = axs[-1],
            node_size=10,
        )

    title = rf"focal beliefs at t={t}, $\omega_0={initial_w}$, $\epsilon={epsV}$, $\mu={mu}$, $\rho={params['social_edge_weight']}${eval(intervention_name)[-1]}"
    print(title)
    # fig.suptitle(title)

    fig.tight_layout()
    return fig, ax


# %%
selected_seeds = seeds
ps = [
    # (0.2, 0.0, 0.0),
    # (0.2, epsV_on, mu_on),
    # (0.4, 0.0, 0.0, 0.0),
    # (0.4, epsV_on, mu_on, lam_on),
    (initial_w, 0.0, 0.0, 0.0),
    (initial_w, epsV_on, mu_on, lam_on),
]
for intervention_name in [
    "noI",
    "weak_focal",
    "strong_focal",
]:  # , "weak_focal", "strong_focal"]:
    params.update(dict(zip(intervention_params, eval(intervention_name))))

    T = 200
    focal_att = "a"
    for initial_w, epsV, mu, lam in ps:
        t = T
        fig, ax = plot_nets_tot(
            t,
            epsV,
            mu,
            lam,
            initial_w,
            selected_seeds,
            intervention_name=intervention_name,
            sorted=(epsV != 0),
        )
        fig.suptitle(
            rf"$\epsilon={epsV}$, $\lambda={lam}$, $\omega_0={initial_w}$; {eval(intervention_name)[-1]}, $t={t}$",
            fontsize=15,
        )
        fig.tight_layout()

        fname = (
            resultsfolder
            + "figs/"
            + f"focalNets_M{params['M']}_epsV{epsV}_mu{mu}_initW{initial_w}_t{t}_rho{params['social_edge_weight']}{generate_filename(params)[1]}.png"
        )
        print(fname)
        plt.savefig(
            fname,
            bbox_inches="tight",
        )

# %%


def plot_nets_befDurAfter(ts, epsV, mu, lam, initial_w, seed, intervention_name):
    params.update(dict(zip(intervention_params, eval(intervention_name))))
    params.update(
        {
            "initial_w": initial_w,
            "epsV": epsV,
            "mu": mu,
            "lam": lam,
        }
    )
    fig, axs = plt.subplots(1, 3, figsize=(12 * 3 / 7, 3))
    # mod = res.loc[
    #     (res.initial_w == params["initial_w"])
    #     & (res.epsV == params["epsV"])
    #     & (res.mu == params["mu"])
    #     & (res.seed==seed),
    # ]
    params["seed"] = seed

    filename = generate_filename(params)[0]
    simOut = pd.read_csv(filename + ".csv", low_memory=False)
    for t, ax in zip(ts, axs.flatten()):
        params["focal_att"] = "a"
        plot_net(
            simOut,
            t,
            params,
            "white",
            ax=ax,
            cbar=False,  ##(ax == axs[1]),
            node_size=10,
        )
        ax.text(
            0.96,
            0.04,
            f"t={t}",
            color="grey",
            fontsize=15,
            va="bottom",
            ha="right",
            transform=ax.transAxes,
        )

    # intervention_string = (
    #     (
    #         "\n"
    #         + ("strong " if params["intervention_strength"] == strong_focal[2] else "weak ")
    #         + "intervention "
    #         + (
    #             "on focal"
    #             if params["intervention_att"] == params["focal_att"]
    #             else "on non-focal"
    #         )
    #     )
    #     if params["intervention_att"] is not None
    #     else "\nno intervention"
    # )
    print(
        rf"focal beliefs, $\omega_0={initial_w}$, $\epsilon={epsV}$, $\mu={mu}$, $\rho={params['social_edge_weight']}${eval(intervention_name)[-1]}"
    )
    fig.suptitle(
        rf"focal beliefs, $\omega_0={initial_w}$, $\epsilon={epsV}$, $\mu={mu}$, $\rho={params['social_edge_weight']}$ {eval(intervention_name)[-1]}"
    )

    fig.tight_layout()
    # fig.text(
    #     0.73,
    #     0.02,
    #     "background color =\n modularity of focal beliefs",
    #     fontsize=7,
    #     ha="left",
    #     va="bottom",
    #     transform=fig.transFigure,
    # )
    return fig, axs


# %%
intervention_name = "strong_focal"
for seed in seeds:
    print(seed)
    for epsV, mu, lam in [(0.0, 0.0, 0.0), (epsV_on, mu_on, lam_on)]:
        fig, axs = plot_nets_befDurAfter(
            [100, 150, 200], epsV, mu, lam, initial_w, seed, intervention_name
        )
        # fig.suptitle(
        #     rf"$\epsilon={epsV}$, $\mu={mu}$, $\omega_0={initial_w}$; {intervention_name}, $t={t}$, seed {seed}"
        # )
        fig.tight_layout()
        fname = (
            resultsfolder
            + "figs/"
            + f"focalNetsBeforeAfter_seed{seed}_M{params['M']}_epsV{epsV}_mu{mu}_initW{initial_w}_t{t}_rho{params['social_edge_weight']}{generate_filename(params)[1]}.png"
        )
        plt.savefig(fname)


# %%
# seed 1-5, initial_w=0.4, weak focal!
# network 0.3, 0.005
# eps=0.5, mu =.1
# rho =0.3
# M=10

# params.update(dict(zip(intervention_params, eval("strong_focal"))))
# params.update(
#     {"initial_w": initial_w, "epsV": epsV_on, "mu": 0.0, "lam": lam_on, "seed": 36}
# )
# plt.figure()
# filename = generate_filename(params)[0]
# simOut = pd.read_csv(filename + ".csv", low_memory=False)
# plt.hist(
#     simOut.loc[simOut.time == 99]["a"],
#     color="red",
#     alpha=0.2,
#     bins=np.linspace(-1, 1, 21),
# )
# plt.hist(
#     simOut.loc[simOut.time == 150]["a"],
#     color="green",
#     alpha=0.2,
#     bins=np.linspace(-1, 1, 21),
# )
# plt.hist(
#     simOut.loc[simOut.time == 190]["a"],
#     color="lightgrey",
#     alpha=0.2,
#     bins=np.linspace(-1, 1, 21),
# )
# plt.hist(
#     simOut.loc[simOut.time == 195]["a"],
#     color="grey",
#     alpha=0.2,
#     bins=np.linspace(-1, 1, 21),
# )
# plt.hist(
#     simOut.loc[simOut.time == 200]["a"],
#     color="k",
#     alpha=0.2,
#     bins=np.linspace(-1, 1, 21),
# )
# %%

# plt.figure()
# times = [0, 99, 149, 200]
# a = pd.DataFrame(
#     np.array([simOut.loc[simOut.time == t]["a"].values for t in times]).T, columns=times
# )
# # a = (100*a).astype("int")
# t1 = 99
# t2 = 200
# heatmap, xedges, yedges = np.histogram2d(a[t1], a[t2], bins=np.linspace(-1, 1, 8))
# cmap = plt.get_cmap("plasma")
# cmap.set_under("white")
# sns.heatmap(
#     heatmap.T,
#     cmap=cmap,
#     vmin=0.1,
#     cbar=True,
#     xticklabels=np.round(xedges[:-1] + 1 / 7, 1),
#     yticklabels=np.round(yedges[:-1] + 1 / 7, 1),
#     annot=True,
# )
# plt.xlabel(f"t = {t1}")
# plt.ylabel(f"t = {t2}")
# %%


# def plot_BN(simOut, i, t, params, intervention_name, scaleE=3, scaleN=100):
#     ag = simOut.loc[(simOut.time == t) & (simOut.agent_id == i)]
#     G = nx.Graph()
#     for a in atts:
#         G.add_node(a, value=ag[a])
#     edgelist = list(combinations(atts, 2))
#     for e in edgelist:
#         G.add_edge(e[0], e[1], value=ag[f"({e[0]},{e[1]})"].values[0])

#     widths = [scaleE * G.edges[e]["value"] for e in edgelist]
#     cmap = plt.get_cmap("coolwarm")
#     norm = plt.Normalize(
#         vmin=-1, vmax=1
#     )  # 2. Convert scalar edge values to RGBA tuples
#     edge_colors = [cmap(norm(G.edges[e]["value"])) for e in edgelist]

#     fig = plt.figure(figsize=(8 / 2.54, 8 / 2.54))
#     ax = plt.axes()
#     pos = nx.circular_layout(G)
#     nx.draw_networkx_edges(
#         G=G, pos=pos, edge_color=edge_colors, width=widths, edgelist=edgelist
#     )
#     node_colors = [cmap(norm(ag[att])) for att in atts]
#     nx.draw_networkx_nodes(
#         G=G, pos=pos, nodelist=atts, node_color=node_colors, node_size=scaleN
#     )
#     nx.draw_networkx_labels(G=G, pos=pos, labels=dict(zip(atts, atts)), font_size=7)
#     ax.plot(
#         pos[focal_att][0],
#         pos[focal_att][1],
#         marker="s",
#         ms=20,
#         markerfacecolor="None",
#         markeredgecolor="grey",
#         markeredgewidth=2,
#         zorder=-1,
#     )
#     ax.text(
#         0.02,
#         0.02,
#         f"t={t}" + "\n" + f"ag {i}",
#         ha="left",
#         va="bottom",
#         transform=ax.transAxes,
#     )

#     ax.text(
#         1,
#         1.0,
#         (
#             f"{'after ' if t>params["intervention_period"][1] else ''} {eval(intervention_name)[-1]}"
#             if t > params["intervention_period"][0]
#             else ("")
#         ),
#         ha="right",
#         va="bottom",
#         transform=ax.transAxes,
#     )
#     ax.text(
#         1,
#         0.98,
#         rf"$\epsilon={params['epsV']}$"
#         + "\n"
#         + rf"$\mu={params['mu']}$,"
#         + "\n"
#         + rf"$\lambda={params['lam']}$",
#         ha="right",
#         va="top",
#         transform=ax.transAxes,
#         fontsize=8,
#     )

#     ax.axis(False)
#     ax.set_facecolor((0, 0, 0, 0))
#     return fig, ax


# %%


def get_stats(simOut, avg=1):
    # Define time windows
    times_before = np.arange(100 - avg + 1, 101)
    times_short_after = np.arange(150 - avg + 1, 151)
    times_long_after = np.arange(200 - avg + 1, 201)

    # Calculate average opinions for each agent in each time window
    ops_before = (
        simOut.loc[simOut.time.isin(times_before)].groupby("agent_id")[focal_att].mean()
    )
    ops_short_after = (
        simOut.loc[simOut.time.isin(times_short_after)]
        .groupby("agent_id")[focal_att]
        .mean()
    )
    ops_long_after = (
        simOut.loc[simOut.time.isin(times_long_after)]
        .groupby("agent_id")[focal_att]
        .mean()
    )

    # Classify agents based on initial opinions
    prior_negs = ops_before[ops_before < 0].index
    prior_pos = ops_before[ops_before > 0].index

    # Count agents with positive opinions in long-term
    longpost_pos = ops_long_after[ops_long_after > 0].index

    # Analyze agents who started negative
    if len(prior_negs) > 0:
        # Agents who flipped to positive short-term
        temporary_flippers = ops_short_after.loc[
            (ops_before < 0) & (ops_short_after > 0)
        ].index
        # Agents who flipped to positive long-term
        longterm_flippers = ops_long_after.loc[
            (ops_before < 0) & (ops_long_after > 0)
        ].index
        # Agents who flipped temporarily but reverted to negative
        reverters = ops_long_after.loc[
            (ops_before < 0) & (ops_short_after > 0) & (ops_long_after < 0)
        ].index
        # Agents who stayed negative throughout
        resistant = ops_long_after.loc[
            (ops_before < 0) & (ops_short_after < 0) & (ops_long_after < 0)
        ].index
    else:
        temporary_flippers = []
        longterm_flippers = []
        reverters = []
        resistant = []

    # Agents who started positive but became negative
    anticompliants = ops_long_after.loc[prior_pos][
        ops_long_after.loc[prior_pos] < 0
    ].index

    return (
        len(longpost_pos),
        len(prior_negs),
        len(longterm_flippers),
        len(reverters),
        len(resistant),
        len(anticompliants),
    )


# %%

res = []
for intervention_name in [
    "noI",
    "weak_focal",
    "medium_focal",
    "strong_focal",
    "xxstrong_focal",
]:
    params.update(dict(zip(intervention_params, eval(intervention_name))))
    for seed in seeds:
        params["seed"] = seed
        for eps, lam in [(0.0, 0.0), (epsV_on, lam_on)]:
            params.update(
                {
                    "initial_w": initial_w,
                    "epsV": eps,
                    "mu": mu_on,
                    "lam": lam,
                }
            )
            filename = generate_filename(params)[0]
            simOut = pd.read_csv(filename + ".csv", low_memory=False)
            (
                n_compliants,
                n_prior_negs,
                n_compliant_flipper,
                n_reverters,
                n_resistant,
                n_falseflippers,
            ) = get_stats(simOut, avg=10)
            res.append(
                [
                    intervention_name,
                    eps,
                    lam,
                    seed,
                    n_compliants,
                    n_prior_negs,
                    n_compliant_flipper,
                    n_reverters,
                    n_resistant,
                    n_falseflippers,
                ]
            )
res = pd.DataFrame(
    res,
    columns=[
        "intervention_name",
        "eps",
        "lam",
        "seed",
        "n_compliants",
        "n_prior_negs",
        "n_compliant_flipper",
        "n_reverters",
        "n_resistant",
        "n_falseflippers",
    ],
)
# %%
# (res.groupby(["intervention_name", "eps"])["n_compliants"] > 90).sum()

# %%
res2 = res.copy()
res2["frac_compliantsFlipper"] = res2["n_compliant_flipper"] / res2["n_prior_negs"]

res2["frac_resilient"] = res2["n_reverters"] / res2["n_prior_negs"]

res2["frac_resistant"] = res2["n_resistant"] / res2["n_prior_negs"]
# %%
fig, ax = plt.subplots(1, 1, figsize=(7 / 2.54, 7 / 2.54))
sns.stripplot(
    res.loc[res.intervention_name == "noI"],
    x="eps",
    y="n_compliants",
    ax=ax,
    hue="intervention_name",
    dodge=True,
    size=3,
    alpha=0.5,
    legend=False,
    clip_on=True,
)

sns.boxplot(
    res.loc[(res.intervention_name == "noI") & (res.eps > 0)],  ## CAREFUL
    hue="intervention_name",
    x="eps",
    y="n_compliants",
    ax=ax,
    legend=False,
    whis=[0, 100],
    boxprops={"alpha": 0.3},
)
ax.set_xticks([0, 1])
ax.set_xticklabels(["fixed", "adaptive"])
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_ylim(-3, 103)
ax.set_xlim(-0.5, 1.5)
ax.vlines(0.5, 0, 100, color="k", lw=0.5)
fig.tight_layout()
plt.savefig(
    resultsfolder + f"figs/count_compliant-resilient-resistant-agents.png", dpi=900
)


# %%

#################################
#####  Fractions   #####
#################################


fig, axs = plt.subplots(1, 3, sharey=True, figsize=(18 / 2.54, 7 / 2.54))
axs[0].set_title("Compliant")
sns.stripplot(
    res2,
    x="eps",
    y="frac_compliantsFlipper",
    ax=axs[0],
    hue="intervention_name",
    dodge=True,
    size=3,
    alpha=0.5,
    legend=False,
    clip_on=False,
)


axs[1].set_title("Resilient")
sns.stripplot(
    res2,
    x="eps",
    y="frac_resilient",
    ax=axs[1],
    hue="intervention_name",
    dodge=True,
    size=3,
    alpha=0.5,
    legend=False,
    clip_on=False,
)

axs[2].set_title("Resistant")
sns.stripplot(
    res2,
    x="eps",
    y="frac_resistant",
    ax=axs[2],
    hue="intervention_name",
    dodge=True,
    size=3,
    alpha=0.5,
    legend=False,
    clip_on=False,
)


sns.boxplot(
    res2,
    hue="intervention_name",
    x="eps",
    y="frac_compliantsFlipper",
    ax=axs[0],
    legend=False,
    whis=[0, 100],
    boxprops={"alpha": 0.3},
)
sns.boxplot(
    res2,
    hue="intervention_name",
    x="eps",
    y="frac_resilient",
    ax=axs[1],
    legend=False,
    whis=[0, 100],
    boxprops={"alpha": 0.3},
)
sns.boxplot(
    res2,
    hue="intervention_name",
    x="eps",
    y="frac_resistant",
    ax=axs[2],
    whis=[0, 100],
    boxprops={"alpha": 0.3},
    legend=False,
)
for ax in axs:
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["fixed", "adaptive"])
    ax.set_xlabel("")
    ax.set_ylim(-0.05, 1.05)
    ax.vlines(0.5, 0, 1, color="k", lw=0.5)
axs[0].set_ylabel("")
fig.tight_layout()
plt.savefig(
    resultsfolder + f"figs/frac_compliantFlipper-resilient-resistant-agents.png",
    dpi=900,
)
# %%

intervention_name = "medium_focal"
params.update(dict(zip(intervention_params, eval(intervention_name))))
seed = 42
params["seed"] = seed
params.update(
    {
        "initial_w": initial_w,
        "epsV": epsV_on,
        "mu": mu_on,
        "lam": lam_on,
    }
)


def get_edges(params, edges, t):
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
        simOut.loc[simOut.time.isin(times_long_after)]
        .groupby("agent_id")[focal_att]
        .mean()
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

    all = simOut.loc[simOut.time == t].set_index("agent_id")
    a = all.loc[:, edges].abs().mean(axis=1)
    aneg = all.loc[prior_negs, edges].abs().mean(axis=1)
    res = all.loc[resistant, edges].abs().mean(axis=1)
    rev = all.loc[reverters, edges].abs().mean(axis=1)
    return a.values, aneg.values, res.values, rev.values


t = 200

focal_edges = [e for e in edge_labels if focal_att in e]

allsFoc = []
anegsFoc = []
revertersFoc = []
resistantsFoc = []
for s in seeds:
    params.update({"seed": s})
    a, aneg, res, rev = get_edges(params, focal_edges, t)
    # allsFoc.append(a)
    anegsFoc.append(aneg)
    revertersFoc.append(rev)
    resistantsFoc.append(res)
# allsFoc = np.concat(allsFoc)
anegsFoc = np.concat(anegsFoc)
revertersFoc = np.concat(revertersFoc)
resistantsFoc = np.concat(resistantsFoc)


nonfocal_edges = [e for e in edge_labels if focal_att not in e]
alls = []
anegs = []
reverters = []
resistants = []
for s in seeds:
    params.update({"seed": s})
    a, aneg, res, rev = get_edges(params, nonfocal_edges, t)
    # alls.append(a)
    anegs.append(aneg)
    reverters.append(rev)
    resistants.append(res)
# alls = np.concat(alls)
anegs = np.concat(anegs)
reverters = np.concat(reverters)
resistants = np.concat(resistants)

# %%
fig, axs = plt.subplots(1, 2, sharey=False, sharex=True, figsize=(16 / 2.54, 8 / 2.54))

bins = np.linspace(0, 2, 21)
sns.histplot(
    anegsFoc,
    color="grey",
    bins=bins,
    label="agents with " + r"$x_{\rm foc}<0$" + "\n" + r"before Intervention",
    stat="count",
    alpha=0.4,
    kde=False,
    ax=axs[0],
    linewidth=0.4,
)
sns.histplot(
    resistantsFoc,
    # color="red",
    bins=bins,
    label="...resistant",
    stat="count",
    alpha=0.7,
    kde=False,
    ax=axs[0],
    linewidth=0.4,
)
sns.histplot(
    revertersFoc,
    # color="green",
    bins=bins,
    label="...resilient",
    stat="count",
    alpha=0.7,
    kde=False,
    ax=axs[0],
    linewidth=0.4,
)
axs[0].set_ylabel("frequency")
sns.histplot(
    anegs,
    label="agents with " + r"$x_{\rm foc}<0$" + "\n" + r"before intervention",
    color="grey",
    bins=bins,
    stat="count",
    alpha=0.4,
    kde=False,
    ax=axs[1],
    linewidth=0.4,
)
sns.histplot(
    resistants,
    # color="red",
    bins=bins,
    label="...resistant",
    stat="count",
    alpha=0.7,
    kde=False,
    ax=axs[1],
    linewidth=0.4,
)
sns.histplot(
    reverters,
    # color="green",
    label="...resilient",
    bins=bins,
    stat="count",
    alpha=0.7,
    kde=False,
    ax=axs[1],
    linewidth=0.4,
)
axs[1].set_ylabel("")
axs[0].set_yticks([])
axs[1].set_yticks([])
axs[0].set_xlim(0, 2)
axs[0].set_xlabel(r"average $|\omega_{{\rm foc}, i}|$" + rf" at $t={t}$")
axs[1].set_xlabel(r"average $|\omega_{i,j}|$" + rf" at $t={t}$")
axs[1].legend(fontsize=8)
fig.tight_layout()


# %%
from scipy.stats import mannwhitneyu, ks_2samp

pairs = [
    ("all vs resistant", allsFoc, resistantsFoc),
    ("all vs resilient", allsFoc, revertersFoc),
    ("resistant vs resilient", resistantsFoc, revertersFoc),
    ("all vs resistant NonFocal", alls, resistants),
    ("all vs resilient NonFocal", alls, reverters),
    ("resistant vs resilient NonFocal", resistants, reverters),
]

for name, a, b in pairs:
    stat, p = ks_2samp(a, b, alternative="greater")
    print(f"{name}: KS={stat:.3f}, p={p:.5f}")

    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    print(f"{name}: MW U test: p={p:.5f}")

# %%
