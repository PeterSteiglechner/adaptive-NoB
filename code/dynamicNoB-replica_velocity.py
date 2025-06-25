# This file contains an extension of the network of beliefs to allow both dynamic node values and dynamic edge weights. 
# Note: the network of beliefs contains only personal belief edges.
# ADD: we now also consider social updating.
# ADD: we now completely copy the NoB
# ADD: we use Hebbian learning via activation

import numpy as np
import pandas as pd
#import pingouin # for partial correlation: pcorr
from itertools import combinations
#import copy
import string
from help_functions import get_socialOmega, initialise_socialnetwork, glauber_probabilities_withSocial
from help_functions import hebbianV, socialinfluence, socialinfluenceMult, decay
import json

#################################
#####  DYNAMIC MODEL   #####
#################################
def update_step(t, focal_att, agent_id, agentdict, atts, Wij, params, **kwargs):
    epsV, eps, mu, lam, dt = params["epsV"], params["eps"], params["mu"], params["lam"], params["dt"]
    socialInfl_type = params["socInfType"]
    network_type = params["socNetType"]
    memory = params["memory"]

    belief_options, beta_pers, beta_soc = params["belief_options"], params["beta_pers"], params["beta_soc"]
    social_edge_weight = params["social_edge_weight"]

    agent = agentdict[agent_id]
    x = agent["x"]
    v = agent["v"]
    belief_network = agent["BN"]

    edgelist = list(belief_network.keys())
    np.random.shuffle(edgelist)
    for edge in edgelist:
        i, j = edge
        weight_ij = belief_network[edge]

        # Decay + Hebbian
        delta_beta = decay(weight_ij, lam) + hebbianV(weight_ij, v[i], v[j], epsV)

        # Social Influence
        if socialInfl_type == "copy":
            if network_type == "observe-all":
                neighbours = list(agentdict.keys())
            else:
                neighbours = agent.get("neighbours", [])
            
            if neighbours:
                sampled_neighbour = np.random.choice(neighbours)
                curr_weight = agentdict[sampled_neighbour]["BN"].get(edge, 0)
            else:
                curr_weight = np.nan
        else:
            curr_weight = 0 if Wij(agent_id).empty else Wij(agent_id).loc[i, j]

        if ~np.isnan(curr_weight): 
            delta_beta += socialinfluence(weight_ij, curr_weight, mu)

        # Update weight with clipping
        # belief_network[edge] = np.clip(weight_ij + dt * delta_beta, -1, 1)
        belief_network[edge] = weight_ij + dt * delta_beta
           
    # NODE UPDATING
    x_prior = dict(zip(atts, [x[a] for a in atts]))
    _atts = list(x.keys())
    np.random.shuffle(_atts)
    for att in _atts:
        if att == focal_att:
            soc_beliefs = [agentdict[ag]["x"][att] for ag in neighbours] 
        else:
            soc_beliefs = []
        # options are M points in -1...1
        ps = glauber_probabilities_withSocial(att, belief_options, x, belief_network, atts, soc_beliefs, social_edge_weight, beta_pers=beta_pers, beta_soc=beta_soc)
        x[att] = np.random.choice(belief_options, p=ps)

    # VELOCITY UPDATING
    agent["velo_past"].append([x[a]- x_prior[a] for a in atts])
    if len(agent["velo_past"]) > memory:
        agent["velo_past"] = agent["velo_past"][1:]
    new_v = np.array(agent["velo_past"]).mean(axis=0)

    # Save updates
    agent["BN"] = belief_network
    agent["x"] = x
    agent["v"] = dict(zip(atts, new_v))

    return agentdict

    
#%% 
#################################
#####  Simulation Run   #####
#################################
def dynSim_NoB(agent_ids, focal_att, atts, params):
    """
    Run a dynamic simulation of adaptive NoB. 

    Opinions are initialised random between -1 and 1. 
    All identities are set to none

    Args:
        agent_ids (list): List of agent identifiers.
        atts (list): Names of belief dimensions (opinion attributes).
        params (dict): Configuration parameters including:
            - "eps" (float): Strength of Hebbian Learning from holding both opinions
            - "epsV" (float): Strength of Hebbian Learning from Activation
            - "mu" (float): Strength of Social Influence
            - "lam" (float): Strength of edge weight regularisation 
            - "socInfType" (str): Type of social influence ("correlation", "co-occurence", "copy").
            - "socNetType" (str): Network structure ("observe-all", "observe-neighbours").
            - "parties" (list): Identity groups, excluding "none".
            - "indegree" (float): In-group connection probability.
            - "outdegree" (float): Out-group connection probability.
            - "initial_w" (float): Initial edge weight in belief network.
            - "seed" (int): Random seed for reproducibility.
            - "T" (float): Total time.
            - "dt" (float): Time step.
            - "track_times" (list): Time steps to track results.
            - "beta_pers" (float): attention to personal dissonance; 1/TempP 
            - "beta_scc" (float): attention for social dissonance; 1/TempS
            - "belief_options" (list): possible belief states
            - "social_edge_weight" (float): fixed edge weight of a social link (for node updating)

    Returns:
        simOut (pd.DataFrame): Final belief network weights and opinions for each agent.
        snapshots (pd.DataFrame): Belief states recorded at specified time steps.
        agent_ids (list): Ordered list of agent IDs (original order).
    """
    np.random.seed(params["seed"])

    # Initialise opinions randomly in [-1, 1]
    opinions = pd.DataFrame(
        [np.random.choice(params["belief_options"], replace=True, size=len(atts)) for agent in agent_ids],
        index=agent_ids, columns=atts
    )

    # Initialise agent network
    assert (len(agent_ids)%2)==0
    group_size = int(len(agent_ids)/2)
    identity = pd.Series(["A"]*group_size + ["B"]*group_size, index=agent_ids)
    agent_dict = initialise_socialnetwork(agent_ids, identity, opinions, params)
    neighbours_dict = {agname: ag.get("neighbours", []) for agname, ag in agent_dict.items()}

    agent_ids = list(agent_dict.keys())
    original_agent_ids = list(agent_ids)  

    print("simulate", end="...")
    time_steps = np.arange(0, params["T"]+1, step=params["dt"])
    
    snapshot = [
                [agent_dict[ag]['BN'][e] for e in agent_dict[ag]['BN']] +
                [agent_dict[ag]['x'][att] for att in atts]
                for ag in original_agent_ids
            ]
    results_over_time = [snapshot]
    
    # Main simulation loop (starts at time[1] to skip t=0)
    for t in time_steps[1:]:
        if t % 10 == 0: print(t, end=", ")

        # Set up social influence matrix if needed
        socInfType = params["socInfType"]
        if socInfType in {"co-occurence", "correlation"}:
            Wij = get_socialOmega(agent_dict, agent_ids, opinions, atts, params)
        else:
            Wij = None

        # Update all agents (in random order)
        np.random.shuffle(agent_ids)
        for ag in agent_ids:
            agent_dict = update_step(t, focal_att, ag, agent_dict, atts, Wij, params)

        # Record results at tracked times
        if t in params["track_times"] or t==time_steps[-1]:
            snapshot = [
                [agent_dict[ag]['BN'][e] for e in agent_dict[ag]['BN']] +
                [agent_dict[ag]['x'][att] for att in atts]
                for ag in original_agent_ids
            ]
            results_over_time.append(snapshot)
    print("done")

    # Create final output DataFrame
    edge_labels = [f"({i},{j})" for i, j in agent_dict[original_agent_ids[0]]['BN']]
    snapshots = []
    for t, res in zip(params["track_times"] , results_over_time):
        snap = pd.DataFrame(
            data=res, # edge weights and beliefs (columns) per agent (row)
            columns = edge_labels + atts, index=original_agent_ids)
        snap.loc[original_agent_ids, "agent_weight"] = pd.Series(np.ones(len(original_agent_ids)), index=original_agent_ids)
        snap.loc[original_agent_ids, "identity"] = identity.loc[original_agent_ids]
        snap["t"] = t
        snap = snap.reset_index(names="agent_id")
        snapshots.append(snap)
    snapshots_df = pd.concat(snapshots)
    final_snap = snapshots[-1]
    final_snap.loc[original_agent_ids, "neighbours"] = [json.dumps(neighbours_dict[ag]) for ag in original_agent_ids]

    return snapshots[-1], snapshots_df, original_agent_ids, neighbours_dict


# %%
#################################
#####  MAIN   #####
#################################
if __name__=="__main__":
    for seed in [0,1,9]:
        T = 100
        params = {
            "seed":seed,
            "n": 100,
            "T":T,
            "dt":1,
            "M": 10, # number of beliefs
            "track_times": np.arange(0,T+1, 1),
            "socNetType":"observe-neighbours",  # observe-neighbours or observe-all
            "socInfType":None,   # correlation or co-occurence or copy
            "eps":None,
            "epsV":None,
            "mu":None, 
            "lam":None,
            "parties": ["A", "B"], 
            "withinClusterP":0.4, 
            "betweenClusterP":0.01,
            "initial_w": 0.4,
            #"belief_jump":0.1,
            "beta_pers":None,
            "beta_soc":None,
            "belief_options": np.linspace(-1,1,7), 
            "social_edge_weight": 1,
            "memory": 2,
        }
        atts = list(string.ascii_lowercase[:params["M"]])
        edgeNames = [f"({i},{j})" for i,j in list(combinations(atts, 2))]
        agentlist = list(range(params["n"]))        
        
        focal_att = "a"

        # base scenario:
        # OLD (0.4,0.2,0.05, Temp, "copy", "observe-neighbours") where Temp = 0.01, 0.1, 1, 10
        #     
        epsV = 1.
        mu = 0.5
        paramCombis = [
            # eps, epsV, mu, lam, beta_pers, beta_soc, socInfType, socNetType
            (0.0, epsV, mu,0.0, 0.5, 0.5, "copy", "observe-neighbours"), 
            (0.0,epsV, mu,0.0, 2, 0.5, "copy", "observe-neighbours"), 
            (0.0, epsV, mu,0.0, 0.5, 2, "copy", "observe-neighbours"), 
            (0.0, epsV, mu,0.0, 2, 2, "copy", "observe-neighbours"), 
            ]
        
        resultsfolder = "results-dynNoB_replica/"
        
        for eps, epsV, mu, lam, beta_pers, beta_soc, socInfType, socNetType in paramCombis:
            params["eps"] = eps
            params["epsV"] = epsV
            params["mu"]  = mu
            params["lam"] = lam
            params["beta_pers"] = beta_pers
            params["beta_soc"] = beta_soc 
            params["socInfType"] = socInfType
            params["socNetType"] = socNetType
            print(eps, epsV, mu, lam, beta_pers, beta_soc, socInfType, socNetType)

            simOut, snapshots, original_agent_ids, neighbours_dict = dynSim_NoB(agentlist, focal_att, atts, params)
                
            socNetName = f"{socNetType}"+(f"(Stoch-{len(params['parties'])}-Block-{params['withinClusterP']}-{params['betweenClusterP']})" if "neighbours" in socNetType else "")
            
            filename = resultsfolder+f"dynamicNoB-P+S_M-{params['M']}_n-{params['n']}_{socInfType}-"+socNetName+f"beta-p{beta_pers}-s{beta_soc}"+f"_epsV{epsV}-m{params['memory']}_eps{eps}_mu{mu}_lam{lam}_initialW-{params['initial_w']}_seed{params['seed']}"
            
            # Store final results
            simOut.to_csv(filename+".csv")

            # Store results over time        
            snapshots.to_csv(filename+"_overTime.csv")

#%%


#################################
#####  VIS   #####
#################################

import matplotlib.pyplot as plt 
import seaborn as sns

import networkx as nx 

#%%

params["initial_w"] = 0.4
beta_pers = 2
beta_soc = 2
eps = 0.0
mu =0.
epsV= 0.


fig = plt.figure()
res_arr = []
belief_observe = "avgbelief"
#seeds_samples = np.random.choice([0,1,9], 3, replace=False)
seeds_samples = [0,1,9]
for s in seeds_samples:
    filename = resultsfolder+f"dynamicNoB-P+S_M-{params['M']}_n-{params['n']}_{socInfType}-"+socNetName+f"beta-p{beta_pers}-s{beta_soc}"+f"_epsV{epsV}-m{params['memory']}_eps{eps}_mu{mu}_lam{lam}_initialW-{params['initial_w']}_seed{s}"
    snapshots = pd.read_csv(filename+"_overTime.csv")
    for t in [T]:
        snapshots.loc[snapshots.t==t, "avgbelief"] = snapshots.loc[snapshots.t==t, atts].mean(axis=1)
    
    #sns.kdeplot(snapshots.loc[snapshots.t.isin([0,T]), [belief_observe, "t"]], x=belief_observe, hue="t", clip=(-1,1), legend=False, palette="coolwarm") 
    res_arr.append(snapshots.loc[snapshots.t==T, belief_observe].values)


ax = fig.add_subplot(111) 
#sns.histplot(snapshots.loc[snapshots.t.isin([0,T]), [belief_observe, "t"]], x=belief_observe, hue="t", bins=np.linspace(-1-1/14,1+1/14, 21), palette="viridis", alpha=0.2 ) 
sns.histplot(res_arr, bins=np.linspace(-1-1/14,1+1/14, 21), palette="Set1", alpha=0.2, legend=True, ax=ax) 
leg = ax.get_legend()
leg.set_title("seed")
#%%


params["initial_w"] = 0.4
beta_pers = 2
beta_soc = 2
eps = 0.0
mu =0.5
epsV= 1

filename = resultsfolder+f"dynamicNoB-P+S_M-{params['M']}_n-{params['n']}_{socInfType}-"+socNetName+f"beta-p{beta_pers}-s{beta_soc}"+f"_epsV{epsV}-m{params['memory']}_eps{eps}_mu{mu}_lam{lam}_initialW-{params['initial_w']}_seed{s}"
snapshots = pd.read_csv(filename+"_overTime.csv")
finalsnap = pd.read_csv(filename+".csv")
neighbours_dict = {ag: json.loads(finalsnap.loc[ag, "neighbours"]) for ag in agentlist}
G = nx.from_dict_of_lists(neighbours_dict)

plt.rcParams.update({"font.size":15})

for t in range(100):
    seed_samples = [0]#np.random.choice(range(10), size=10, replace=False)
    for s in seed_samples:
        fig = plt.figure()
        ax = plt.axes()
        colors = [(snapshots.loc[(snapshots.t==t) & (snapshots.agent_id==i), focal_att]) for i in neighbours_dict.keys()]
        pos = nx.spring_layout(G, seed=1) if t>0 else pos
        nx.draw_networkx_edges(G, pos=pos, width=0.5, alpha=0.5)
        nx.draw_networkx_nodes(G, pos=pos, node_color=colors, cmap="coolwarm", vmax=1, vmin=-1, node_size=100)
        ax.set_title(fr"$\beta_p = {beta_pers}, \beta_s={beta_soc}, \epsilon={epsV}, \mu={mu}$")
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin = -1, vmax=1))
        sm._A = []
        plt.colorbar(sm, label="focal attitude", ticks=[-1,0,1], ax=ax)
        ax.annotate("agents", pos[38], (pos[38][0]+0.3, pos[38][1]-0.3), fontsize=15, color="grey",  arrowprops=dict( arrowstyle="-", connectionstyle="arc3,rad=0.2",color="grey", shrinkA=5, shrinkB=5))
        ax.annotate("", pos[56], (pos[38][0]+0.4, pos[38][1]-0.3), fontsize=15, color="grey", arrowprops=dict( arrowstyle="-", connectionstyle="arc3,rad=-0.2",color="grey", shrinkA=5, shrinkB=5))
        ax.annotate("", pos[58], (pos[38][0]+0.5, pos[38][1]-0.3), fontsize=15, color="grey", arrowprops=dict( arrowstyle="-", connectionstyle="arc3,rad=-0.2",color="grey", shrinkA=5, shrinkB=5))
        ax.text(0.05,0.05,f"$t={t}$", transform=ax.transAxes, va="bottom", ha="left", fontsize=15, color="grey")
        plt.savefig(f"figs_gif_CA-workshop/socNet_epsV-{epsV}-m{params['memory']}_mu{mu}_lam{lam}_initialW-{params['initial_w']}_beta-{beta_pers}-{beta_soc}_seed{s}_t{t:03d}.png")
#%%

epsV = 1
mu = 0.5
beta_pers = 2
beta_soc = 2
s = 0

filename = resultsfolder+f"dynamicNoB-P+S_M-{params['M']}_n-{params['n']}_{socInfType}-"+socNetName+f"beta-p{beta_pers}-s{beta_soc}"+f"_epsV{epsV}-m{params['memory']}_eps{eps}_mu{mu}_lam{lam}_initialW-{params['initial_w']}_seed{s}"

snapshots = pd.read_csv(filename+"_overTime.csv")

selected_edgeNames = ["(a,b)", "(d,e)"]
fig, axs = plt.subplots(1,2, sharey=True, sharex=True)
sns.histplot(snapshots.loc[(snapshots.t==T) & (snapshots.identity=="A")][selected_edgeNames], legend=False, ax = axs[0], bins=np.linspace(-1.5,1.5, 12))
sns.histplot(snapshots.loc[(snapshots.t==T) & (snapshots.identity=="B")][selected_edgeNames], legend=True, ax=axs[1] , bins=np.linspace(-1.5,1.5, 12))
axs[0].set_title("group A")
axs[1].set_title("group B")


fig.suptitle("edge weights \n"+fr"$\beta_p = {beta_pers}$, $\beta_s={beta_soc}$"+"\n"+fr"and $\epsilon = {epsV}$, $\mu={mu}$ (colour=edge)")
fig.tight_layout()

# %%
diff = snapshots.loc[(snapshots.t==T) & (snapshots.identity=="A")][edgeNames].mean(axis=0) - snapshots.loc[(snapshots.t==T) & (snapshots.identity=="B")][edgeNames].mean(axis=0)

fig, ax = plt.subplots(1,1)
sns.histplot(diff, ax=ax)
ax.set_xlabel("Group differences in edge weights")
ax.set_ylabel("")
ax.set_yticks([])


#%% 
e_A =snapshots.loc[(snapshots.t==T) & (snapshots.identity=="A")][edgeNames]
e_B = snapshots.loc[(snapshots.t==T) & (snapshots.identity=="B")][edgeNames] 
inds = (e_A.mean()-e_B.mean()).sort_values().index

e_B_long = e_B.loc[:, inds].melt(var_name='Belief Network Edges', value_name='Edge Weight')
e_B_long['Group'] = 'B'
e_A_long = e_A.loc[:, inds].melt(var_name='Belief Network Edges', value_name='Edge Weight')
e_A_long['Group'] = 'A'
df = pd.concat([e_A_long, e_B_long])

ax = plt.axes() 
sns.boxplot(data=df, ax=ax, x='Belief Network Edges', y='Edge Weight', hue='Group', whis=[0, 100], palette={'A': 'purple', 'B': 'green'})
ax.legend(title='Group')
ax.set_xticklabels([])
plt.tight_layout()


#%%

diff = snapshots.loc[(snapshots.t==T) & (snapshots.identity=="A")][edgeNames].mean(axis=0) - snapshots.loc[(snapshots.t==T) & (snapshots.identity=="B")][edgeNames].mean(axis=0)

diff[]

#%%



from help_functions import social_energy, energy
social_energy(colors[0], [colors[nb].values[0] for nb in neighbours_dict[0]], 1)
i = 0
belief_network = {
            (att1, att2): 1 for att1, att2 in combinations(atts, 2)
        }
energy(snapshots.loc[(snapshots.t==T) & (snapshots.agent_id==i), atts], "a", atts, belief_network )
# %%
# let's look at an individual agent network
import matplotlib as mpl

epsV =0.
mu = 0.
beta_pers = 2
beta_soc = 2
s = 0

filename = resultsfolder+f"dynamicNoB-P+S_M-{params['M']}_n-{params['n']}_{socInfType}-"+socNetName+f"beta-p{beta_pers}-s{beta_soc}"+f"_epsV{epsV}-m{params['memory']}_eps{eps}_mu{mu}_lam{lam}_initialW-{params['initial_w']}_seed{s}"

snapshots = pd.read_csv(filename+"_overTime.csv")


i = np.random.choice(agentlist)
i= 48
print(i)

ag = snapshots.loc[(snapshots.t==T) & (snapshots.agent_id == i)]
G = nx.Graph()
for a in atts:
    G.add_node(a, value=ag[a])
for na, a in enumerate(atts):
    for b in atts[:na]:
        G.add_edge(a,b, value=ag[f"({b},{a})"].values[0])
pos = nx.circular_layout(G)
edgelist = list(G.edges())
edgewidths = [3*abs(G.edges[e]["value"]) for e in G.edges()]
edgecol = [plt.get_cmap("coolwarm")( mpl.colors.Normalize(vmin=-1.5, vmax=1.5)(G.edges[e]["value"])) for e in G.edges()]
nx.draw_networkx_edges(G, pos, width=edgewidths, edge_color=edgecol, edgelist=edgelist, edge_cmap="viridis", edge_vmax=1.5, edge_vmin=-1.5)
nodelist = list(G.nodes())
nodecol = [G.nodes[n]["value"] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=nodecol, cmap="coolwarm", vmax=1, vmin=-1)
plt.gca().text(pos[focal_att][0]+0.1, pos[focal_att][1], "focal", rotation=90, fontsize=20)
plt.gca().text(0.99,0.99, f"group {ag.identity.values[0]}", va="top", ha="right", transform=plt.gca().transAxes, fontsize=20)


plt.axis("off")

# %%
ag = snapshots.loc[(snapshots.t==T), edgeNames+["agent_id"]]

# %%
from scipy.spatial.distance import pdist, squareform
import pandas as pd

df = snapshots.loc[(snapshots.t==T),  edgeNames+["agent_id"]]
# Compute pairwise Frobenius (Euclidean) distances
distances = pdist(df[edgeNames].values, metric='euclidean')  # condensed distance matrix
dist_matrix = pd.DataFrame(squareform(distances), 
                           index=df.agent_id, 
                           columns=df.agent_id)
dist_matrix["ag1"] = dist_matrix.columns
dists = dist_matrix.melt(var_name="ag2", value_name="Frobenius_distance", id_vars="ag1")
dists.loc[dists["Frobenius_distance"].argmax()]



# %%

from analysis_help import plot_network
from netgraph import Graph


epsV =1
mu = 0.5
beta_pers = 2
beta_soc = 2
s = 0

filename = resultsfolder+f"dynamicNoB-P+S_M-{params['M']}_n-{params['n']}_{socInfType}-"+socNetName+f"beta-p{beta_pers}-s{beta_soc}"+f"_epsV{epsV}-m{params['memory']}_eps{eps}_mu{mu}_lam{lam}_initialW-{params['initial_w']}_seed{s}"

snapshots = pd.read_csv(filename+"_overTime.csv")

i = 82
ag = snapshots.loc[(snapshots.t==T) & (snapshots.agent_id == i)]
G = nx.Graph()
for a in atts:
    G.add_node(a, value=ag[a])
for na, a in enumerate(atts):
    for b in atts[:na]:
        G.add_edge(a,b, value=ag[f"({b},{a})"].values[0])
bn_adj = pd.Series({(b, a): ag[f"({b},{a})"].iloc[0] for na, a in enumerate(atts) for b in atts[:na]})
scaleE = 3
widths = (bn_adj*scaleE).to_dict()
#widths = {k: (v if abs(v) else 0) for k, v in widths.items()}
edge_labels = False
node_labels = dict(zip(G.nodes(), list(G.nodes())))
import matplotlib.cm as cm
import matplotlib.colors as mcolors
cmap = cm.get_cmap('coolwarm')
norm = mcolors.Normalize(vmin=-1, vmax=1)  # Adjust based on your data

# 2. Convert scalar edge values to RGBA tuples
edge_colors = {tuple(sorted((a, b))): cmap(norm(v)) for (a, b), v in bn_adj.items()}
ax = plt.axes()


pos = nx.spring_layout(G, weight='value', seed=41)  # seed for reproducibility
Graph(
    G,
    node_layout = pos,#_kwargs=dict(edge_lengths = (0.001 / (abs(bn_adj) + 1e-6)).to_dict()),
    node_size=5,
    node_shape="o",
    node_color="gainsboro",
    node_edge_color="w",
    edge_width=widths,
    edge_color=edge_colors,     # <-- pass a dict of float values
    edge_cmap=plt.get_cmap("coolwarm"),            # <-- colormap to map floats to RGBA
    edge_vmin=-1.5,                    # <-- adjust these based on your data
    edge_vmax=1.5,
    edge_layout="curved",
    edge_labels=False,
    node_labels=node_labels,
    node_label_offset=0.,
    node_label_fontdict={"fontsize": 12},
    ax=ax, 
)
    

# %%


atts = ["climate", "gender", "migration", "inequality", "environment", "health", "taxes"]
G = nx.Graph()
ag = {a: None for a in atts}  # Placeholder values
for a in atts:
    G.add_node(a, value=ag[a])

# Add edges with weights representing typical associations (positive or negative)
edges = [
    ("climate", "environment", 0.8),
    ("climate", "health", 0.2),
    ("climate", "taxes", -0.6),
    ("climate", "inequality", 0.3),

    ("environment", "health", 0.6),
    ("environment", "taxes", -0.2),
    ("environment", "inequality", 0.4),

    ("health", "inequality", 0.7),
    ("health", "taxes", -0.3),

    ("taxes", "inequality", 0.5),
    ("taxes", "migration", -0.5),
    ("taxes", "gender", -0.),

    ("inequality", "gender", 0.6),
    ("inequality", "migration", 0.5),

    ("migration", "gender", 0.),
    ("migration", "health", -0.),

    ("gender", "health", 0.),
]

# Add to graph
for u, v, w in edges:
    G.add_edge(u, v, weight=w)

import matplotlib.cm as cm
import matplotlib.colors as mcolors
cmap = cm.get_cmap('coolwarm')
norm = mcolors.Normalize(vmin=-1, vmax=1)  # Adjust based on your data

# 2. Convert scalar edge values to RGBA tuples
edge_colors =({(u,v): cmap(norm(w["weight"])) for u,v, w in (list(G.edges(data=True)))})

fig, ax = plt.subplots(1,1,figsize=(20/2.54, 6/2.54))
Graph(
    G,
    node_layout_kwargs=dict(edge_lengths = {(u,v): 0.1 / (abs(w["weight"])+ 1e-6) for u,v, w in list(G.edges(data=True))}), 
    node_size=7,
    node_shape="o",
    node_color="gainsboro",
    node_edge_color="w",
    edge_width=({(u,v): 10 * w["weight"] for u,v, w in (list(G.edges(data=True)))}),
    edge_color=edge_colors,     # <-- pass a dict of float values
    edge_cmap=plt.get_cmap("coolwarm"),            # <-- colormap to map floats to RGBA
    edge_vmin=-1.,                    # <-- adjust these based on your data
    edge_vmax=1.,
    edge_layout="curved",
    edge_labels=False,
    node_labels=dict(zip(G.nodes(), atts)),
    node_label_offset=0.,
    node_label_fontdict={"fontsize": 8},
)

    
# %%
