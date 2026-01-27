# %%
import matplotlib as mpl 
import numpy as np
import pandas as pd
from itertools import combinations
import string
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import netCDF4
from matplotlib import patches
import json
from scipy.spatial.distance import pdist, squareform
# from sklearn.manifold import MDS
import xarray as xr

from matplotlib.gridspec import GridSpec


sns.set_style("ticks", {"axes.linewidth": 0.5})
smallfs = 8
bigfs = 9
plt.rc("font", family="sans-serif")
plt.rc("font", size=smallfs)  # Ticklabels, legend labels, etc.
plt.rc("axes", labelsize=bigfs)  # Axis labels
plt.rc("axes", titlesize=bigfs)  # Titles
plt.rcParams.update({"font.size": bigfs})
plt.rcParams.update({"axes.titlesize": bigfs})
plt.rcParams.update({"axes.labelsize": bigfs})
plt.rcParams.update({"legend.fontsize": smallfs})


# %%
resultsfolder = "2025-12-29/"

ds = xr.load_dataset("processed_data/2025-12-29_modelAdaptiveBN_baselineConfig_results_detailed_withBeliefsAndEdges.ncdf", engine="netcdf4")


#################################
#####  #################################
#####  #################################
#####  VISUALISE   #####
#################################   #####
#################################   #####
#################################


# %%
belief_dimensions = list(ds.belief.values)
edgeNames = list(ds.edge.values)
focal_dim = ds.attrs["focal_dim"]
eval_time = ds.attrs["evaluation_time_for_response"]
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
cat_type = pd.CategoricalDtype(categories=responses, ordered=True)
svals = [0, 16]

# cmap = dict(
#     zip(
#         responses,
#         ["#7fc97f", "#fdc086", "#386cb0", "#beaed4", "#f0027f", "#bf5b17", "#666666"],
#     )
# )
cmap = dict(
    zip(
            responses+["NA"],
            ["#4DAF4A", "#A6D854", "#469FDB", "#B8DFF3", "#7A0177", "#E41A1C",  "#666666"],
        )
    )
# %%

#################################
#####  Time Series All   #####
#################################


def plot_timeseries(df_pivot, final_values, T, t, window, dim, ext_pressure_strength):
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
            if dim == focal_dim
            else (f"belief ${dim}$" if len(dim) == 1 else f"belief relation ${dim}$")
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
    if len(dim) == 1:
        ax_main.set_ylim(-1, 1)
    ax_main.set_clip_on(False)
    if ext_pressure_strength > 0:
        # ["#640000", "#850000", "#B20000", "#DE0000", "#FF0000"]
        int_colors = dict(zip([1, 2, 4, 8, 16], [0.1, 0.175, 0.25, 0.325, 0.4]))
        y0, y1 = ax_main.get_ylim()
        xx = [100, 149]
        xx = [ttt for ttt in xx if ttt <= t]
        if len(xx) > 0:
            ax_main.fill_between(
                xx,
                [y0] * len(xx),
                [y1] * len(xx),
                color="red",
                alpha=int_colors[ext_pressure_strength],
                zorder=-1,
                lw=0,
            )
            ax_main.text(
                (xx[0]+xx[-1])/2,
                y0 + (y1 - y0) / 2,
                rf"external"+"\n"+rf"pressure" + "\n" + r"$s="+rf"{ext_pressure_strength}$",
                fontsize=smallfs,
                va="center",
                ha="center",
                color="red",
            )

    # Histogram on the right
    ax_hist = fig.add_subplot(gs[1], sharey=ax_main)
    y0, y1 = ax_main.get_ylim()
    bins = np.linspace(-1.001, 1.001, 21) if len(dim) == 1 else np.linspace(y0, y1, 21)
    sns.histplot(
        final_values,
        y="belief_smooth",
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

dims = list(ds.belief.values)


def plot_BN_ax(ag_b, ag_e, ax, intStart=100, intEnd=150, scaleE=1, scaleN=100, highlightfocal=True, layout="spring"):
    G = nx.Graph()
    for a in dims:
        G.add_node(a, value=ag_b.loc[a, "belief_value"])
    edgelist = list(combinations(dims, 2))
    for e in edgelist:
        G.add_edge(e[0], e[1], value=ag_e.loc[f"{e[0]}_{e[1]}", "edge_weight"])
    widths = [scaleE * G.edges[e]["value"] for e in edgelist]
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-1, vmax=1)
    edge_colors = [cmap(norm(G.edges[e]["value"])) for e in edgelist]
    if layout=="circle":
        pos = nx.circular_layout(G)    
    else:
        pos = nx.spring_layout(G, weight="value", seed=1, k=0.9)
    nx.draw_networkx_edges(
        G=G, pos=pos, edge_color=edge_colors, width=widths, edgelist=edgelist, ax=ax
    )
    node_colors = [cmap(norm(ag_b.loc[dim, "belief_value"])) for dim in dims]
    nx.draw_networkx_nodes(
        G=G, pos=pos, nodelist=dims, node_color=node_colors, node_size=scaleN, ax=ax
    )
    nodelistMod = [dim for dim in dims if abs(ag_b.loc[dim, "belief_value"])<0.7]
    nodelistEx =  [dim for dim in dims if abs(ag_b.loc[dim, "belief_value"])>=0.7]
    nodelabelsMod = [dim if not dim==focal_dim else r"foc" for dim in nodelistMod]
    nodelabelsEx = [dim if not dim==focal_dim else r"foc" for dim in nodelistEx]
    nx.draw_networkx_labels(
        G=G, pos=pos, labels=dict(zip(nodelistEx, nodelabelsEx)) , font_size=7, font_color="whitesmoke", ax=ax
    )
    nx.draw_networkx_labels(
        G=G, pos=pos,  labels=dict(zip(nodelistMod, nodelabelsMod)), font_size=7, font_color="k", ax=ax
    )
    if highlightfocal:
        ax.plot(
            pos[focal_dim][0],
            pos[focal_dim][1],
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
    y0,yh= ax.get_ylim()
    ax.set_ylim(y0*1.1, yh*1.1)
    x0,xh= ax.get_xlim()
    ax.set_xlim(x0*1.1, xh*1.1)
    return ax


def plot_timeseries_ag(
    df_pivot, final_values, ag_b, ag_e, i, T, t, window, dim, ext_pressure_strength, adaptive
):
    fig = plt.figure(figsize=(16 / 2.54, 5 / 2.54))
    gs = GridSpec(1, 3, width_ratios=[4, 1, 1.5], wspace=0.02)

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
            if dim == focal_dim
            else (f"belief ${dim}$" if len(dim) == 1 else f"belief relation ${dim}$")
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
    if len(dim) == 1:
        ax_main.set_ylim(-1, 1)
    ax_main. set_clip_on(False)
    if ext_pressure_strength > 0 and t > 100:
        # ["#640000", "#850000", "#B20000", "#DE0000", "#FF0000"]
        int_colors = dict(zip([1, 2, 4, 8, 16], [0.1, 0.175, 0.25, 0.325, 0.4]))
        y0, y1 = ax_main.get_ylim()
        xx = [100, 150]
        xx = [ttt for ttt in xx if ttt <= t]
        if len(xx) > 0:
            ax_main.fill_between(
                xx,
                [y0] * len(xx),
                [y1] * len(xx),
                color="red",
                alpha=int_colors[ext_pressure_strength],
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
    bins = np.linspace(-1.001, 1.001, 21) if len(dims) == 1 else np.linspace(y0, y1, 21)
    sns.histplot(
        final_values,
        y="belief_smooth",
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
    print( int(adaptive)==1, t!=0 )
    plot_BN_ax(
        ag_b,
        ag_e,
        ax_net,
        intStart=100,
        intEnd=150,
        layout="spring" if int(adaptive)==1 and t!=0 else "circle"
    )
    fig.subplots_adjust(right=0.99, bottom=0.25)
    return


# %%

#################################
#####  LINEPLOT + DENSITY   #####
#################################
dim = "0"
adaptive = False
ext_pressure_strength = 0

seed = 0  # np.random.randint(10)
window = 0
df = (
    ds.sel(adaptive=adaptive, seed=seed, s_ext=ext_pressure_strength, belief=dim)[
        "belief_value"
    ]
    .to_dataframe()
    .reset_index()
)
df = df.sort_values(["agent_id", "time"])
t = 200
final_values = df.loc[df.time == t, ["agent_id", "belief_value"]]
if window > 0:
    df["belief_smooth"] = df.groupby("agent_id")["belief_value"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
else:
    df["belief_smooth"] = df["belief_value"]
df_pivot = df.pivot(index="time", columns="agent_id", values="belief_smooth")

# %%
times = [0]  # np.arange(0, 201, 1)
for t in times:
    final_values = df.loc[df.time == t, ["agent_id", "belief_smooth"]]
    plot_timeseries(df_pivot, final_values, 200, t, window, dim, ext_pressure_strength)
    if len(times) > 10:
        plt.savefig(
            resultsfolder
            + f"_gifs/{seed}/all/t{t}_{'fixed_' if not adaptive else ''}_sext-{ext_pressure_strength}_spring.png",
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
adaptive = True
for ag_i in [90]:  # [8, 90, 92]:
    ag = ds.sel(adaptive=adaptive, s_ext=ext_pressure_strength, seed=seed, agent_id=ag_i)[
        ["belief_value"]
    ].to_dataframe()
    ag_edges = ds.sel(
        adaptive=adaptive, s_ext=ext_pressure_strength, seed=seed, agent_id=ag_i
    )[["edge_weight"]].to_dataframe()

    times = [100]  # np.arange(0, 201, 1)
    for t in times:
        ag_b_t = ag.reset_index().loc[ag.reset_index().time == t].set_index("belief")
        ag_e_t = (
            ag_edges.reset_index()
            .loc[ag_edges.reset_index().time == t]
            .set_index("edge")
        )
        final_values = df.loc[df.time == t, ["agent_id", "belief_smooth"]]
        plot_timeseries_ag(
            df_pivot,
            final_values,
            ag_b_t,
            ag_e_t,
            ag_i,
            200,
            t,
            window,
            dim,
            ext_pressure_strength,
            adaptive=adaptive
        )
        if len(times) > 10:
            plt.savefig(
                resultsfolder
                + f"_gifs/{seed}/{ag_i}/t{t}_sext-{ext_pressure_strength}_spring_FOCUS.png",
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

# np.arange(0, 201, 1)
if len(times) > 10:
    for ag_i, ext_pressure_strength, adaptive, seed in combis:  # len(times) == 200:
        print(ag_i, ext_pressure_strength, adaptive, seed)
        images = []
        for t in times:
            im = Image.open(
                resultsfolder
                + "_gifs/"
                + f"{seed}/{ag_i}/t{t}_{'fixed_' if not adaptive else ''}_sext-{ext_pressure_strength}_spring.png"
            )
            images.append(im)
        last_frame = len(images)
        for x in range(0, 9):
            im = images[last_frame - 1]
            images.append(im)
        gifname = (
            "figs/"
            + f"snaps_{'fixed' if not adaptive else f'{ds.attrs['epsilon']}-{ds.attrs['lambda']}'}_sext-{ext_pressure_strength}_w0-{ds.attrs['initial_w']}_seed{seed}_ag{ag_i}_t0-200_spring.gif"
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
# ----------------------------------------------
# -------    SIMULATIONS
# ----------------------------------------------

T0 = 0
T = 100

# ".", "c0", "c200", ".", "d0", "d200"

ext_pressure_strength = 0
seed = 1
window=0
for adaptive in [False, True]:
    fig, axs = plt.subplot_mosaic([["t", "hist", ".", f"a{T0}", ".", f"a{T}", ], ["t", "hist", ".",  f"b{T0}", ".", f"b{T}"], [".", ".", ".",  f"b{T0}", ".", f"b{T}"]], 
                              width_ratios=[0.5,0.1, 0.02, 0.2,0.06, 0.2], height_ratios=[1,0.7,0.3], figsize=(16/2.54, 6/2.54))

    dim = focal_dim
    df = (
        ds.sel(adaptive=adaptive, seed=seed, s_ext=ext_pressure_strength, belief=dim)[
            "belief_value"
        ]
        .to_dataframe()
        .reset_index()
    )
    df = df.sort_values(["agent_id", "time"])
    t = T
    final_values = df.loc[df.time == t, ["agent_id", "belief_value"]]
    if window > 0:
        df["belief_smooth"] = df.groupby("agent_id")["belief_value"].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    else:
        df["belief_smooth"] = df["belief_value"]
    df_pivot = df.pivot(index="time", columns="agent_id", values="belief_smooth")


    colors=plt.get_cmap("Set1")
    ags = [98, 9] # 9 is great.
    for n, ag_i in enumerate(ags):  # [8, 90, 92]:
        ag = ds.sel(adaptive=adaptive, s_ext=ext_pressure_strength, seed=seed, agent_id=ag_i)[
            ["belief_value"]
        ].to_dataframe()
        ag_edges = ds.sel(
            adaptive=adaptive, s_ext=ext_pressure_strength, seed=seed, agent_id=ag_i 
        )[["edge_weight"]].to_dataframe()

        ag_b_t = ag.reset_index().loc[ag.reset_index().time == t].set_index("belief")
        ag_e_t = (
            ag_edges.reset_index()
            .loc[ag_edges.reset_index().time == t]
            .set_index("edge")
        )
        final_values = df.loc[df.time == t, ["agent_id", "belief_smooth"]]


        df_pivot = df_pivot.loc[df_pivot.index <= T]
        if n==0:
            axs["t"].plot([],[],lw=1, alpha=0.4, label="agent", color="grey")
            leg= axs["t"].legend(loc="center left")
            df_pivot.plot(ax = axs["t"], lw=0.3, alpha=0.2, legend=False, color="grey")
            axs["t"].set_xlabel("time")
            axs["t"].set_ylabel("focal belief")
            axs["t"].set_clip_on(False)
            axs["t"].set_xlim(0, T)
            axs["t"].set_ylim(-1.0, 1.0)
            axs["t"].set_yticks([-1,0,1])
            ax_hist = axs["hist"]
            ax_hist.sharey(axs["t"])
            y0, y1 = axs["t"].get_ylim()
            bins = np.linspace(-1.001, 1.001, 21) if len(dims) == 1 else np.linspace(y0, y1, 21)
            sns.histplot(
                final_values,
                y="belief_smooth",
                bins=bins,
                orientation="horizontal",
                color="grey",
                edgecolor="black",
                ax=ax_hist,
            )
            # --- Color bars with coolwarm ---
            cmap = mpl.cm.coolwarm
            norm = mpl.colors.Normalize(vmin=y0, vmax=y1)

            for patch in ax_hist.patches:
                # For horizontal bars, y is the vertical position
                y = patch.get_y() + patch.get_height() / 2
                patch.set_facecolor(cmap(norm(y)))

            ax_hist.set_ylabel("")
            ax_hist.grid(False)
            ax_hist.set_clip_on(False)
            ax_hist.axis("off")
            # ax_hist.text(
            #     1,
            #     0.5,
            #     f"Histogram\nat $t={t}$",
            #     ha="right",
            #     va="center",
            #     # rotation=270,
            #     fontsize=smallfs,
            #     transform=ax_hist.transAxes,
            # )
        
        df_pivot.T.loc[ag_i].plot(ax=axs["t"], lw=3, ls='-', color=colors(n+2), alpha=1, legend=False, clip_on=False)
        for t in [T0,T]:

            ag_b_t = ag.reset_index().loc[ag.reset_index().time == t].set_index("belief")
            ag_e_t = (
                    ag_edges.reset_index()
                    .loc[ag_edges.reset_index().time == t]
                    .set_index("edge")
                )
            ax_net = axs[f"{string.ascii_lowercase[n]}{t}"]
            plot_BN_ax(
                ag_b_t,
                ag_e_t,
                ax_net,
                intStart=100,
                intEnd=150,
                scaleN=100,
                scaleE=1.5,
                highlightfocal=False, 
                layout="spring" if int(adaptive) and t>0 else "circle",
            )
    for l in ["a","b"]:
        # ConnectionPatch handles the transform internally so no need to get fig.transFigure
        arrow = patches.ConnectionPatch(
            (1,0.5),
            (0,0.5),
            coordsA=axs[l+f"{T0}"].transAxes,
            coordsB=axs[l+f"{T}"].transAxes,
            # Default shrink parameter is 0 so can be omitted
            color="black",
            arrowstyle="-|>",  # "normal" arrow
            mutation_scale=10,  # controls arrow head size
            linewidth=1,
        )
        axs[l+f"{T0}"].text(1.07,0.02,rf"$t={T0}$", ha="right", va="bottom", transform=axs[l+f"{T0}"].transAxes, fontsize=smallfs, rotation=40)
        axs[l+f"{T}"].text(1.07,0.02,rf"$t={T}$", ha="right", va="bottom", transform=axs[l+f"{T}"].transAxes, fontsize=smallfs, rotation=40)
        fig.patches.append(arrow)

    fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.98,bottom=0.02, left=0.07, right=0.99)

    for n, (i, y) in enumerate(zip(ags, [0.93, 0.43])):
        bboxprops = dict(
                boxstyle="round",
                facecolor=colors(n+2),
                edgecolor=colors(n+2),
                alpha=0.3,
            )
        fig.text(0.77,y, f"agent {i}", fontdict={"fontsize":smallfs, "bbox":bboxprops})
    # fig.set_facecolor("pink")
    import string
    for n, ax in enumerate([axs["t"], axs[f"a{T0}"], axs[f"b{T0}"]]): 
        ax.text(-0.12 if n==0 else 0, 0.975, string.ascii_uppercase[n], fontsize=12, fontdict={"weight":"bold"},va="top", ha="left", transform=ax.transAxes)

    print(np.sign(final_values)["belief_smooth"].value_counts())
    plt.savefig(f"figs/2026-01-21_examplesim_{'adaptive' if adaptive else 'fixed'}.png", dpi=600)
# %%

# %%
