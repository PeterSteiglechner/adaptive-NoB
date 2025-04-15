import numpy as np
import pandas as pd
import pingouin
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import networkx as nx
from netgraph import Graph
import matplotlib as mpl
from model_equ import *
smallfs = 7
bigfs = 9
supersmallfs =6
plt.rcParams.update({"font.size":smallfs})


# Functions

weighted_avg = lambda df, weightcol, col: (df[weightcol] * df[col]).sum(axis="index", skipna=True)/df[weightcol].sum(axis="index")

weighted_std = lambda df, weightcol, col: np.nan if all(df[col].isna()) else ((df[weightcol] * (df[col]-df[col].mean())**2).sum(axis="index", skipna=True)/df[weightcol].sum(axis="index"))**0.5

def get_bn(simOut, edgeNames, edgeWeightThreshold=0.0):
    bn_adj = pd.Series([weighted_avg(simOut, "agent_weight", e) for e in edgeNames], index=edgeNames, name="weight")
    bn_adj = bn_adj.rename(index=dict(zip(edgeNames, [(e.split("↔")[0],  e.split("↔")[1]) for e in edgeNames])))
    index = pd.MultiIndex.from_tuples(list(bn_adj.index), names=['source', 'target'])
    edges = pd.Series(bn_adj, index=index, name="weight")
    edges = edges.reset_index()
    edges.loc[abs(edges["weight"])<edgeWeightThreshold, "weight"] = edgeWeightThreshold * np.sign(bn_adj.loc[abs(bn_adj)<edgeWeightThreshold])
    bn_adj.loc[abs(bn_adj)<edgeWeightThreshold] = edgeWeightThreshold
    if any(edges["weight"]<0):
        print("negative edges")
        print(edges.loc[edges["weight"]<0])
    return bn_adj, edges

def plot_network(ax, G, bn_adj,  pos,  minEdgeLen=0.01, scaleE=20, scaleN=6, edgelabels=True):
    widths = (bn_adj*scaleE).to_dict()
    edge_labels = dict(zip(widths.keys(), [f"{bn_adj[e]:.1f}" for e in widths.keys()])) if edgelabels else False
    node_labels = dict(zip(G.nodes(), list(G.nodes())))
    print(node_labels)
    if pos=="geometric":
        Graph(G, node_layout=pos, node_size=scaleN, node_layout_kwargs=dict(edge_length=(minEdgeLen/abs(bn_adj)).to_dict()),
            node_shape="o", node_color="gainsboro", node_edge_color="w", #node_edge_width=0.,
            edge_width=widths, edge_layout="curved", 
            edge_labels=edge_labels, edge_label_fontdict={"fontsize":supersmallfs, }, edge_label_position=0.4, edge_cmap="coolwarm", #edge_layout_kwargs={"rad":2},
            node_labels=node_labels, node_label_offset=0., node_label_fontdict={"fontsize":smallfs},
            ax=ax
        )
    else:
        Graph(G, node_layout=pos, node_size=scaleN,
            node_shape="o", node_color="gainsboro", node_edge_color="w", #node_edge_width=0.,
            edge_width=widths, edge_layout="curved", 
            edge_labels=edge_labels, edge_label_fontdict={"fontsize":supersmallfs, }, edge_label_position=0.3, edge_cmap="coolwarm", #edge_layout_kwargs={"rad":2},
            node_labels=node_labels, node_label_offset=0., node_label_fontdict={"fontsize":smallfs},
            ax=ax
        )
    return ax

def plot_centrality(ax, simOut, color="grey"):
    bns = simOut[[f"({e[0]},{e[1]})" for e in list(combinations(atts, 2))]]
    centrality = pd.DataFrame([pd.Series([np.sum([abs(w) for e, w in bn.to_dict().items() if var in e]) for var in atts], index=atts) for i, bn in bns.iterrows()])#.mean()
    beliefs = list(centrality.columns)
    centrality["weight"] = simOut["weight"].values
    pd.DataFrame({
        'mean': [weighted_avg(centrality, "weight", var) for var in beliefs],
        'std': [weighted_std(centrality, "weight", var) for var in beliefs]
    }, index=beliefs).plot(kind='bar', y="mean", yerr='std', alpha=1, ax=ax, color=color, legend=False, rot=0)
    #ax.set_xticklabels(ax.get_xticks(), fontsize=smallfs, rotation=0)
    ax.set_yticks([0,1,2])
    return ax


def plotResults_average(simOut, params, posOption, wave, country, shortcuts, dataname="ESS"):
    eps,mu,lam = params
    bn_adj, edges = get_bn(simOut.loc[:], edgeWeightThreshold=0.01)
    G = nx.from_pandas_edgelist(edges, source='source', target='target', edge_attr='weight', create_using=nx.Graph())
    fig, axs = plt.subplots(2,1, figsize=(9/2.54, 9/2.54), gridspec_kw={"height_ratios":[3,1]})
    ax = plot_network(axs[0], G, bn_adj, posOption, minEdgeLen=0.01)

    ax = plot_centrality(axs[1], simOut) 
    ax.set_ylabel(f"belief centrality")
    
    fig.text(0.5, 0.02, ", ".join([f"{sc}:{att}" for att, sc in shortcuts.items()]), transform=fig.transFigure, va="bottom", ha="center", fontsize=supersmallfs)
    fig.suptitle(rf"Simulated BNs {country}, {dataname} wave {wave} (avg):"+"\n"+rf"$\epsilon={eps}$, $\mu={mu}$, $\lambda={lam}$", fontsize=smallfs)
    return fig, axs, G

def plotResults_parties(simOut, params, parties, country, wave, positionOption, partycols, dataname="ESS", **kwargs):
    eps, mu, lam = params
    fig, axs = plt.subplots(2,len(parties), figsize=(18/2.54,7/2.54), gridspec_kw={"height_ratios":[1.5,1]})
    positions =  {"circular":"circular", "geometric": "geometric", "spring":"spring", "kamada_kawai":"kamada_kawai", "fixed":kwargs["fixedPos"]}
    for p, axCol in zip(parties, range(len(parties))):
        ax = axs[0,axCol]
        bn_adj, edges = get_bn(simOut.loc[simOut.identity==p,:], edgeWeightThreshold=0.01)
        G = nx.from_pandas_edgelist(edges, source='source', target='target', edge_attr='weight', create_using=nx.Graph())
        pos = positions[positionOption]
        plot_network(ax, G, bn_adj, pos, minEdgeLen=0.01, scaleE=20, scaleN=6, edgelabels=False)
        ax.set_title(f"{p}"+"\n"+fr"(n={len(simOut.loc[simOut.identity==p])})", fontsize=smallfs)
        
        ax= axs[1,axCol]
        plot_centrality(ax, simOut.loc[simOut.identity==p, :], color=partycols[p])
        if not (axCol==0):
            ax.set_yticks([0,1,2])
            ax.set_yticklabels([])
        ax.set_ylim(0,2)
    axs[1,0].set_yticklabels(axs[1,0].get_yticks(), fontsize=smallfs)
    axs[1,0].set_ylabel("belief centrality")
    fig.text(0.5, 0.02, ", ".join([f"{sc}:{att}" for att, sc in shortcuts.items()]), transform=fig.transFigure, va="bottom", ha="center", fontsize=supersmallfs)
    fig.suptitle(rf"Simulated BNs {country}, {dataname} wave {wave}  (group avg); $\epsilon={eps}$, $\mu={mu}$, $\lambda={lam}$", )
    return fig, axs
    
def plot_circle_histogram(ax, df, columns, n_bins=20, scale=100, bin_lims=[-1,1], cmap="tab10"):
    colors = sns.color_palette(cmap, n_colors=len(columns))
    bin_edges = np.linspace(bin_lims[0], bin_lims[1], n_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for idx, (col, color) in enumerate(zip(df.columns, colors)):
        hist, _ = np.histogram(df[col].dropna(), bins=bin_edges)
        max_size = scale  # Maximum circle size
        sizes = (hist / hist.sum()) * max_size
        ax.scatter( [idx] * len(bin_centers), bin_centers,  
                   s=sizes, alpha=0.6, color=color, label=col)
    return ax

def custom_boxplot_mean(ax, data, x, y, hue=None, palette="Set2", dodge=True, order=None, hue_order=None, legend=True, tot_width=0.67):
    """
    Custom boxplot-like visualization with mean as the center, variance as the box, and 5th/95th percentiles as whiskers.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        data (pd.DataFrame): The data frame containing the data.
        x (str): Column name for the x-axis categories.
        y (str): Column name for the y-axis values.
        hue (str): Column name for the hue categories (optional).
        palette (str or list): Color palette for the hues.
        dodge (bool): Whether to dodge the hue categories.
        fliersize (float): Size of the outlier markers.
        order (list): Order of the x-axis categories.
        hue_order (list): Order of the hue categories.
        legend (bool): Whether to show the legend.

    Returns:
        ax (matplotlib.axes.Axes): The axis with the plot.
    """
    import matplotlib.patches as patches
    import seaborn as sns

    # Prepare the data
    if order is None:
        order = data[x].unique()
    if hue and hue_order is None:
        hue_order = data[hue].unique()
    colors = sns.color_palette(palette, n_colors=len(hue_order) if hue else 1)

    # Group data by x and hue (if provided)
    grouped = data.groupby([x, hue]) if hue else data.groupby(x)

    # Plot each group
    for i, category in enumerate(order):
        width = tot_width/len(hue_order)
        for j, hue_category in enumerate(hue_order if hue else [None]):
            if hue:
                group = grouped.get_group((category, hue_category))
            else:
                group = grouped.get_group(category)

            # Calculate statistics
            mean = group[y].mean()
            std = group[y].std()
            p5, p95 = group[y].quantile([0.05, 0.95])

            # Calculate positions
            x_pos = -0.33 + i + (j * width if dodge else 0) 
            color = colors[j] if hue else colors[0]

            # Draw the box (mean ± std)
            ax.add_patch(patches.Rectangle((x_pos, mean - std), width, 2 * std, color=color, alpha=0.7))

            # Draw the whiskers (5th and 95th percentiles)
            ax.plot([x_pos+width/2, x_pos+width/2], [p5, mean - std], color=color, alpha=0.7)
            ax.plot([x_pos+width/2, x_pos+width/2], [mean + std, p95], color=color, alpha=0.7)

            # Draw the mean as a horizontal line
            ax.plot([x_pos, x_pos + width], [mean, mean], color="black", linewidth=1.5)
            #ax.plot([x_pos + width/2], [mean], color="black", marker="D", lw=0)

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")

    # Add legend if required

    if legend and hue:
        handles = [patches.Patch(color=colors[i], label=hue_category) for i, hue_category in enumerate(hue_order)]
        leg = ax.legend(handles=handles, title=hue, bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        leg = None
    return ax, leg



def plot_results_over_time(filename, simOut, agentlistOrig, e, partyColsDict, country, ax):
    sims = pd.read_csv(filename+"_overTime.csv")
    sampledata = sims.pivot_table(index="ID", columns="t", values=e).sample(30).T
    ids = list(sampledata.columns)
    cols = simOut.set_index("index").loc[ids, "identity"].map(partyColsDict[country])
    cols = cols.fillna("grey")
    sims.pivot_table(index="ID", columns="t", values=e).mean().T.plot(alpha=1, lw=3, label="", legend=False, ax=ax) # marker="o", markersize=1,
    ax.set_title(e)
    sampledata.plot(alpha=0.2, lw=0.4, label="",  legend=False, ax=ax, color=cols) # marker="o", markersize=1,
    sims["identity"] = [simOut.loc[simOut["index"]==id, "identity"].values for id in agentlistOrig]*int(len(sims)/len(agentlistOrig))
    
    a = sims.pivot_table(index="ID", columns="t", values=e).loc[simOut["index"]]

    a["identity"] =  simOut.set_index("index").loc[agentlistOrig, "identity"]
    
    a = a.groupby("identity").mean().T
    cols = ["grey" if p=="none" or p=="other" or p==np.nan else partyColsDict[country][p] for p in a.columns]
    a.plot(alpha=0.5, lw=3, label="", legend=False, ax=ax, color=cols)
    
    #sims.groupby("identity")[e]#.pivot_table(index="ID", columns="t", values=e).mean().T.plot(alpha=1, lw=3, label="", marker="o", markersize=1, legend=False)
    return ax

