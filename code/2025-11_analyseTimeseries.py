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

from matplotlib.gridspec import GridSpec


sns.set_style("ticks", {"axes.linewidth": 0.5})
smallfs = 8
bigfs = 9
plt.rc("font", family="sans-serif")
plt.rc("font", size=smallfs)  # Ticklabels, legend labels, etc.
plt.rc("axes", labelsize=bigfs)  # Axis labels
plt.rc("axes", titlesize=bigfs)  # Titles
# plt.rcParams.update({"font.size":10})

# %%
epsV_on = 1.0
mu_on = 0.0
lam_on = 0.005
rho = 1.0 / 3.0
initial_ws = [0.2]
initial_w = initial_ws[0]
# %%

resultsfolder = "2025-11-21/"

T = 300
params = {
    "n": 100,
    "belief_options": np.linspace(-1, 1, 21),
    "rho": rho,
    "beta": 3.0,
    "memory": 1,
    "M": 10,
    "starBN": False,
    "depolarisation": False,
    "focal_att": "a",
    "initial_w": initial_w,
    "epsV": None,
    "mu": mu_on,
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
xxxstrong_focal = (range(100, 150), "a", 16, 1, "strong intervention on focal")
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
        f"_lam{params['lam']}_rho{params['rho']:.2f}_beta{params['beta']}_initialW-{params['initial_w']}"
        f"{intervention}"
        f"_seed{params['seed']}"
    ), intervention


# %%
interventions = [
    "noI",
    # "weak_focal",
    "medium_focal",
    # "strong_focal",
    # "xxstrong_focal",
    "xxxstrong_focal",
]
seeds = list(range(10))
conditions = [(0.0, 0.0), (1.0, 0.005)]
Nruns = len(interventions) * len(seeds) * len(conditions)

# ===== allocate final arrays =====
# all aggregated variables will be added here
run_results = {}

vars_to_extract = atts + [e for e in edge_labels]  # if focal_att in e]

# run metadata arrays
meta_seed = np.zeros(Nruns, dtype=int)
meta_adaptive = np.zeros(Nruns, dtype=bool)
meta_int_strength = np.zeros(Nruns, dtype=float)
meta_int_name = np.empty(Nruns, dtype=object)

# ===== main loop =====
run_idx = 0

params["initial_w"] = initial_w
params["mu"] = mu_on

times = np.arange(0, 301, 1)
for intervention_name in interventions:

    params.update(dict(zip(intervention_params, eval(intervention_name))))

    for seed in seeds:

        params["seed"] = seed

        for epsV, lam in conditions:

            # metadata
            meta_seed[run_idx] = seed
            meta_adaptive[run_idx] = epsV > 0
            # meta_int_name[run_idx] = intervention_name

            params["epsV"] = epsV
            params["lam"] = lam

            # load CSV
            filename = generate_filename(params)[0]
            df = pd.read_csv(filename + ".csv", low_memory=False)

            # get strength from file
            if intervention_name == "noI":
                meta_int_strength[run_idx] = 0.0
            else:
                meta_int_strength[run_idx] = df.intervention_strength.dropna().unique()[
                    0
                ]
            # ---- aggregate desired variable (fast) ----
            agg = {
                f"{v}": df.loc[df.time.isin(times)]
                .pivot_table(index="time", columns="agent_id", values=v)
                .values
                for v in vars_to_extract
            }

            # store results in per-column lists
            for k, arr in agg.items():
                if k not in run_results:
                    run_results[k] = np.zeros(
                        (Nruns, len(times), params["n"]), dtype=np.float32
                    )
                run_results[k][run_idx] = arr
            run_idx += 1
            if run_idx % 10 == 0:
                print(
                    run_idx, " of ", len(interventions) * len(seeds) * len(conditions)
                )

# %%

ds = xr.Dataset(
    {k: (("run", "time", "agent_id"), v) for k, v in run_results.items()},
    coords={
        "run": np.arange(Nruns),
        "agent_id": range(params["n"]),
        "seed": ("run", meta_seed),
        "adaptive": ("run", meta_adaptive),
        "intervention_strength": ("run", meta_int_strength),
        "intervention_name": ("run", meta_int_name),
        "time": times,
    },
)

ds = ds.set_index(run=["adaptive", "seed", "intervention_strength"])
ds = ds.unstack("run")
# %%
#################################
#####  Plot 2D hist   #####
#################################
s = 3  # np.random.randint(10)
a = (
    ds.sel(adaptive=True, seed=s, intervention_strength=0)["a"]
    .to_dataframe()
    .reset_index()
)
value_bins = np.arange(-1.05, 1.051, 0.3)  # 21 edges → 20 bins
bins = pd.cut(a["a"], bins=value_bins)
hist_df = pd.crosstab(
    index=bins, columns=a["time"]  # rows = value bins  # cols = time
).rename(
    index=dict(
        zip(
            bins.cat.categories,
            [f"{v:.1f}" for v in (value_bins[:-1] + 0.5 * np.diff(value_bins))],
        )
    )
)
fig, ax = plt.subplots(1, 1)
cmap = plt.get_cmap("viridis")
cmap.set_under("whitesmoke")
sns.heatmap(hist_df, cmap=cmap, vmax=100, vmin=0.1)
# %%

#################################
#####  Time Series All   #####
#################################


def plot_timeseries(df_pivot, final_values, T, t, window, att):
    fig = plt.figure(figsize=(12 / 2.54, 5 / 2.54))
    gs = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.03)

    # Main line plot
    ax_main = fig.add_subplot(gs[0])
    df_pivot = df_pivot.loc[df_pivot.index <= t]
    df_pivot.plot(ax=ax_main, lw=0.5, alpha=0.4, legend=False)
    ax_main.set_xlabel("time")
    ax_main.set_ylabel(
        (
            "focal belief"
            if att == focal_att
            else (f"belief ${att}$" if len(att) == 1 else f"belief relation ${att}$")
        ),
        fontsize=bigfs,
    )
    if window != 0:
        ax_main.text(
            0.99,
            1.01,
            f"smoothed, ${window}$ steps",
            fontsize=smallfs,
            va="bottom",
            ha="right",
            transform=ax_main.transAxes,
        )
    ax_main.text(
        0,
        1.01,
        "fixed belief networks" if not adaptive else "adaptive belief networks",
        fontsize=bigfs,
        va="bottom",
        ha="left",
        transform=ax_main.transAxes,
    )
    ax_main.set_xlim(0, T)
    if len(att) == 1:
        ax_main.set_ylim(-1, 1)
    ax_main.set_clip_on(False)
    if int_s > 0:
        # ["#640000", "#850000", "#B20000", "#DE0000", "#FF0000"]
        int_colors = dict(zip([1, 2, 4, 8, 16], [0.1, 0.175, 0.25, 0.325, 0.4]))
        y0, y1 = ax_main.get_ylim()
        xx = params["intervention_period"]
        xx = [ttt for ttt in xx if ttt <= t]
        if len(xx) > 0:
            ax_main.fill_between(
                xx,
                [y0] * len(xx),
                [y1] * len(xx),
                color="red",
                alpha=int_colors[int_s],
                zorder=-1,
                lw=0,
            )
            ax_main.text(
                xx[0],
                y0 + (y1 - y0) / 2,
                rf"external event" + "\n" + rf"=strength ${int_s}$=",
                fontsize=smallfs,
                va="center",
                ha="left",
                color="red",
            )

    # Histogram on the right
    ax_hist = fig.add_subplot(gs[1], sharey=ax_main)
    y0, y1 = ax_main.get_ylim()
    bins = np.linspace(-1.001, 1.001, 21) if len(atts) == 1 else np.linspace(y0, y1, 21)
    sns.histplot(
        final_values,
        y=att,
        bins=bins,
        orientation="horizontal",
        color="grey",
        edgecolor="black",
        ax=ax_hist,
    )
    # ax_hist.set_xlabel("count", fontsize=bigfs)
    ax_hist.set_ylabel("")
    # ax_hist.set_yticklabels([])
    ax_hist.grid(False)
    ax_hist.set_clip_on(False)
    ax_hist.axis("off")
    ax_hist.text(
        1,
        0.5,
        f"Histogram\nat $t={t}$",
        ha="right",
        va="center",
        # rotation=270,
        fontsize=smallfs,
        transform=ax_hist.transAxes,
    )
    fig.subplots_adjust(right=0.99, bottom=0.25)
    return


# %%

#################################
#####  Time Series one agent   #####
#################################


def plot_BN_ax(ag, ax, intStart=100, intEnd=150, scaleE=1, scaleN=100):
    G = nx.Graph()
    for a in atts:
        G.add_node(a, value=ag[a])
    edgelist = list(combinations(atts, 2))
    for e in edgelist:
        G.add_edge(e[0], e[1], value=ag[f"({e[0]},{e[1]})"])
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
    if intStart is not None:
        if t > intStart and t <= intEnd:
            ax.text(
                0.5,
                1,
                f"external influence",
                va="top",
                ha="center",
                transform=ax.transAxes,
                color="orange",
            )
    ax.axis(False)
    ax.set_facecolor((0, 0, 0, 0))
    return ax


def plot_timeseries_ag(df_pivot, final_values, ag, i, T, t, window, att):
    fig = plt.figure(figsize=(16 / 2.54, 5 / 2.54))
    gs = GridSpec(1, 3, width_ratios=[4, 1, 1.5], wspace=0.03)

    # Main line plot
    ax_main = fig.add_subplot(gs[0])
    df_pivot = df_pivot.loc[df_pivot.index <= t]
    df_pivot.plot(ax=ax_main, lw=0.5, alpha=0.4, legend=False, color="grey")
    df_pivot.T.loc[i].plot(ax=ax_main, lw=4, color="red", alpha=0.4, legend=False)
    ax_main.plot(
        t,
        df_pivot.T.loc[i].loc[t],
        marker="o",
        markersize=5,
        color="red",
        alpha=0.7,
        label="_",
    )
    ax_main.set_xlabel("time")
    ax_main.set_ylabel(
        (
            "focal belief"
            if att == focal_att
            else (f"belief ${att}$" if len(att) == 1 else f"belief relation ${att}$")
        ),
        fontsize=bigfs,
    )
    # if window != 0:
    #     ax_main.text(
    #         0.99,
    #         1.01,
    #         f"smoothed, ${window}$ steps",
    #         fontsize=smallfs,
    #         va="bottom",
    #         ha="right",
    #         transform=ax_main.transAxes,
    #     )
    # ax_main.text(
    #     0,
    #     1.01,
    #     "fixed belief networks" if not adaptive else "adaptive belief networks",
    #     fontsize=bigfs,
    #     va="bottom",
    #     ha="left",
    #     transform=ax_main.transAxes,
    # )
    ax_main.set_xlim(0, T)
    if len(att) == 1:
        ax_main.set_ylim(-1, 1)
    ax_main.set_clip_on(False)
    if int_s > 0 and t > 100:
        # ["#640000", "#850000", "#B20000", "#DE0000", "#FF0000"]
        int_colors = dict(zip([1, 2, 4, 8, 16], [0.1, 0.175, 0.25, 0.325, 0.4]))
        y0, y1 = ax_main.get_ylim()
        xx = params["intervention_period"]
        xx = [ttt for ttt in xx if ttt <= t]
        if len(xx) > 0:
            ax_main.fill_between(
                xx,
                [y0] * len(xx),
                [y1] * len(xx),
                color="red",
                alpha=int_colors[int_s],
                zorder=-1,
                lw=0,
            )
            # ax_main.text(
            #     xx[0],
            #     y0 + (y1 - y0) / 2,
            #     rf"external event" + "\n" + rf"=strength ${int_s}$=",
            #     fontsize=smallfs,
            #     va="center",
            #     ha="left",
            #     color="red",
            # )

    # Histogram on the right
    ax_hist = fig.add_subplot(gs[1], sharey=ax_main)
    y0, y1 = ax_main.get_ylim()
    bins = np.linspace(-1.001, 1.001, 21) if len(atts) == 1 else np.linspace(y0, y1, 21)
    sns.histplot(
        final_values,
        y=att,
        bins=bins,
        orientation="horizontal",
        color="grey",
        edgecolor="black",
        ax=ax_hist,
    )
    # ax_hist.set_xlabel("count", fontsize=bigfs)
    ax_hist.set_ylabel("")
    # ax_hist.set_yticklabels([])
    ax_hist.grid(False)
    ax_hist.set_clip_on(False)
    ax_hist.axis("off")
    ax_hist.text(
        1,
        0.5,
        f"Histogram\nat $t={t}$",
        ha="right",
        va="center",
        # rotation=270,
        fontsize=smallfs,
        transform=ax_hist.transAxes,
    )

    ax_net = fig.add_subplot(gs[2])
    ag_t = ag.loc[t]
    plot_BN_ax(
        ag_t,
        ax_net,
        intStart=100,
        intEnd=150,
    )
    fig.subplots_adjust(right=0.99, bottom=0.25)
    return


# %%

#################################
#####  LINEPLOT + DENSITY   #####
#################################
att = "a"
adaptive = True
int_s = 2

s = 3  # np.random.randint(10)
window = 5
df = (
    ds.sel(adaptive=adaptive, seed=s, intervention_strength=int_s)[att]
    .to_dataframe()
    .reset_index()
)
df = df.sort_values(["agent_id", "time"])
t = 200
final_values = df.loc[df.time == t, ["agent_id", att]]
if window > 0:
    df["att_smooth"] = df.groupby("agent_id")[att].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
else:
    df["att_smooth"] = df[att]
df_pivot = df.pivot(index="time", columns="agent_id", values="att_smooth")

# %%
times = np.arange(0, 201, 1)
for t in times:
    final_values = df.loc[df.time == t, ["agent_id", att]]
    plot_timeseries(df_pivot, final_values, 200, t, window, att)
    plt.savefig(
        resultsfolder
        + f"_gifs/{s}/all/t{t}_{'fixed_' if not adaptive else ''}int{int_s}_spring.png",
        dpi=200,
    )
    plt.close()
    if t % 25 == 0:
        print(t, end=", ")


# %%
# agent 8 for seed 3 int_s 2 is a great conformist
# agent 90 is resilient
# 13 is anti-conformist
# agent 92 is a great resistant
for ag_i in [90]:  # [8, 90, 92]:
    ag = ds.sel(adaptive=True, intervention_strength=int_s, seed=s, agent_id=ag_i)[
        atts + edge_labels
    ].to_dataframe()

    times = [200]  # np.arange(0, 201, 1)
    for t in times:
        final_values = df.loc[df.time == t, ["agent_id", att]]
        plot_timeseries_ag(df_pivot, final_values, ag, ag_i, 200, t, window, att)
        plt.savefig(
            resultsfolder + f"_gifs/{s}/{ag_i}/t{t}_int{int_s}_spring_FOCUS.png",
            dpi=200,
        )
        plt.close()
        if t % 25 == 0:
            print(t, end=", ")
            if t == 200:
                print("")


# %%

from PIL import Image

combis = (
    # ("all", 0, False, 3),
    ("all", 2, False, 6),
    # ("all", 0, True, 3),
    # ("all", 2, True, 3),
    # (8, 2, True, 3),
    # (90, 2, True, 3),
    # (92, 2, True, 3),
)

for ag_i, int_s, adaptive, s in combis:  # len(times) == 200:
    print(ag_i, int_s, adaptive, s)
    images = []
    for t in np.arange(0, 201, 1):
        im = Image.open(
            resultsfolder
            + "_gifs/"
            + f"{s}/{ag_i}/t{t}_{'fixed_' if not adaptive else ''}int{int_s}_spring.png"
        )
        images.append(im)
    last_frame = len(images)
    for x in range(0, 9):
        im = images[last_frame - 1]
        images.append(im)
    gifname = (
        resultsfolder
        + "figsGraz/"
        + f"snaps_{'fixed' if not adaptive else f'{epsV_on}-{mu_on}'}_int{int_s}_w0-{initial_w}_seed{s}_ag{ag_i}_t0-200_spring.gif"
    )
    images[0].save(
        gifname,
        save_all=True,
        append_images=images[1:],
        optimize=True,
        duration=30 if ag_i == "all" else 50,
        loop=0,
    )
    print(gifname)

# %%
