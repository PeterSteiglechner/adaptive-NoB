
#%% 

import numpy as np
import pandas as pd
import networkx as nx
import pingouin  # for partial correlation: pcorr
from itertools import combinations
import string
from help_functions import *

countryXdataset = {"gesis":"Germany", "liss":"Netherlands", "autnes":"Austria"}
atts_datasets = {
    "gesis": ["econ", "migr", "assim", "clim", "euro", "femin"],
    "liss": ["euth", "inequ", "migrAss", "eu", "nr_mig", "asyl"],
    "autnes":[]
}
parties = {
    "Germany": ["Linke", "Grüne", "SPD", "FDP", "CDU/CSU", "AfD", "none"],
    "Netherlands": ["far left", "centre left", "centre", "centre right", "far right", "none"],
    "Austria": ["Grüne", "SPÖ", "NEOS", "ÖVP", "FPÖ", "none"]
}


#################################
#####  DYNAMIC MODEL   #####
#################################

def update_step(t, ag, socNet, atts, Wij, params, **kwargs):
    eps, mu, lam = (params["eps"], params["mu"], params["lam"])
    x = socNet.nodes[ag]['x']
    G_ag = socNet.nodes[ag]["G"]
    
    #  EDGES - Hebbian + Social + Decay
    edgelist = list(G_ag.keys())
    for n_edge in np.random.choice(range(len(edgelist)), size=len(edgelist), replace=False):      
        e = edgelist[n_edge]
        i,j = e
        wij = G_ag[e]
        # Decay
        delBeta = decay(wij, lam)
        # Hebbian
        delBeta += hebbian(wij, x[i], x[j], eps)
        # Social
        if params["socInfType"]=="copy":
            currWij = socNet.nodes[np.random.choice(list(socNet[ag].keys()))]["G"][e]
        else:
            currWij = Wij(ag).loc[i,j]
        delBeta += socialinfluence(wij, currWij, mu)  

        G_ag[e] = np.clip(wij + params["dt"] * delBeta, a_min=-1, a_max=1)

    # TODO insert here node updating. 

    # 

    # SUMMARISE
    socNet.nodes[ag].update({"G": G_ag, "x": x})

    if t in params["track_times"]:
        edge_weights = np.array([G_ag[e] for e in edgelist])
        coherence_values = edge_weights * np.array([x[e[0]] * x[e[1]] for e in edgelist])    
        coherence_dict = {n: coherence_values[np.array([n in e for e in edgelist])].sum() for n in atts}
        coherence_node = [coherence_dict[n] for n in atts]
        socNet.nodes[ag].update({
            "coherence": np.sum(coherence_node),
            "nodeCentrality": [sum(abs(G_ag[e]) for e in edgelist if n in e) for n in atts]
        })
    return socNet

#%%
#################################
#####  Initialise Network   #####
#################################
def initialise_socialnetwork(agentlist, identity, ops, atts, params ):
    print("initialise_socialnetwork")
    np.random.seed(params["seed"])
    if params["socInfType"]=="correlation":
        if params["socNetType"] == "observe-all":
            socNet = nx.empty_graph(agentlist)
        else:
            print("ERROR: This will break. Complete graph of 1000s of nodes")
            quit()
            socNet = nx.complete_graph(agentlist)
    elif (params["socInfType"]=="co-occurence") or (params["socInfType"]=="copy") :
        if params["socNetType"] == "observe-all":
            socNet = nx.empty_graph(agentlist)
        elif params["socNetType"] == "observe-neighbours":            
            sizes = (identity.value_counts().loc[params["parties"]]).astype(int)
            p = [[min(1, params["indegree"]/sizes.loc[s]) if r==s else params["outdegree"]/sizes.sum() for r in params["parties"]]  for s in params["parties"]]
            agentlist_sorted = []
            for party in params["parties"]:
                agentlist_sorted.extend([ag for ag in agentlist if identity.loc[ag]==party])
            socNet = nx.stochastic_block_model(sizes=sizes, p=p, nodelist=agentlist_sorted)
            agentlist = agentlist_sorted         

    for ag in agentlist:
        G_ag = nx.complete_graph(atts)
        G_ag.remove_edges_from(nx.selfloop_edges(G_ag))
        nx.set_edge_attributes(G_ag, 0, "weight")
        socNet.nodes[ag]["x"] =  {att: ops.loc[ag, att] for att in atts} 
        socNet.nodes[ag]["identity"] = identity.loc[ag]
        G_ag_dict = {e: G_ag.edges[e]["weight"] for e in G_ag.edges()}
        socNet.nodes[ag]["G"] = G_ag_dict
        socNet.nodes[ag]["coherence"] = 0
        socNet.nodes[ag]["nodeCentrality"] = [np.sum([G_ag_dict[e] for e in G_ag.edges if (n in e)]) for n in atts]
    
    neighbours = {ag: list(socNet.neighbors(ag)) if params["socNetType"]=="observe-neighbours" else list(socNet.nodes())  for ag in socNet} 

    return socNet, neighbours
     
    

#%% 
#################################
#####  Simulation Run   #####
#################################

def dynSim(atts, wave, country, params, folder, dataset):
    np.random.seed(params["seed"])
    
    filepath = f"{folder}{dataset}-{country.lower()}.csv"
    print(filepath)
    agentlist, weights, ops, identity = initialise(params["n"], atts, wave, filepath)
    print(f"Nr of agents: {len(agentlist)}")


    socNet, neighbours = initialise_socialnetwork(agentlist, identity, ops, atts, params)
    agentlist = list(socNet.nodes())

    if (params["socInfType"]=="co-occurence") or (params["socInfType"]=="correlation"):
        Wij = get_socialOmega(agentlist, ops, neighbours, atts, params)
    else:
        Wij = None

    print("simulate", end="...")
    time = np.arange(0,params["T"], step=params["dt"])
    for t in time[1:]:
        if (t%10==0): 
            print(t, end=", ")
        np.random.shuffle(agentlist)
        for ag in agentlist:
            socNet = update_step(t, ag, socNet, atts, Wij, params)
        #if t in params["track_times"]:
        #    coherenceArr.append([socNet.nodes[ag]['coherence'] for ag in agentlist])
        #    nodeCentralityArr.append([socNet.nodes[ag]['nodeCentrality'] for ag in agentlist])
    print("done")

    edgelist = [f"({e[0]},{e[1]})" for e in socNet.nodes[agentlist[0]]['G'].keys()]
    simOut = pd.DataFrame(
        data=[[socNet.nodes[ag]['G'][e] for e in socNet.nodes[ag]['G'].keys()] + list(ops.loc[ag, atts]) for ag in agentlist], # x and w's (columns) per agent (row)
        columns = edgelist + atts, index=agentlist)
    simOut.loc[agentlist, "agent_weight"] = weights[agentlist]
    simOut.loc[agentlist, "identity"] = identity[agentlist]
    simOut = simOut.reset_index()
    return socNet, simOut


# %%
#################################
#####  MAIN   #####
#################################
if __name__=="__main__":
    folder = "inputdata" #"~/csh-research/projects/opinion-data-curation/data/clean/"
    dataset = "gesis"
    country= countryXdataset[dataset]
    atts = atts_datasets[dataset]
    edgeNames = [f"({i},{j})" for i,j in list(combinations(atts, 2))]
    params={
        "n": 100,    # integer or "all"
        "seed":1,
        "T":100,
        "dt":1,
        "track_times":[0,100],
        "socNetType":"observe-all",  # observe-neighbours or observe-all
        "socInfType":"co-occurence",   # correlation or co-occurence or copy(TODO)
        "eps":None, #filled later
        "mu":None, 
        "lam":None,
        "parties": parties[country], 
        "indegree":10, 
        "outdegree":3,
    }
    # waves defines a set of waves before and after 2021
    waves = {
        "gesis":[list(range(1,20)), list(range(20,28))], 
        "liss":[
            [string.ascii_lowercase[y-8-1] for y in [16,17,18,19,20]], 
            [string.ascii_lowercase[y-8-1] for y in [21,22,23,24]]
            ], 
        "autnes":[],
    }

    paramCombis = [(0.0,0.2,0.05), (0.4,0.0,0.05), (0.4,0.2,0.05)]
    
    for eps, mu, lam in paramCombis:
        params["eps"] = eps
        params["mu"]  = mu
        params["lam"] = lam
        for wave in waves[dataset]:
            socNet, simOut = dynSim(atts, wave, country, params, folder=folder, dataset=dataset)
            waveName = wave if type(wave)==int else f"{wave[0]}-{wave[-1]}"
            filename = f"results/inferBNs-dynamic_{params['socInfType']}-{params['socNetType']}_{dataset.upper()}{waveName}-{country}_eps{eps}_mu{mu}_lam{lam}.csv"
            simOut.to_csv(filename)

            print(eps,mu,lam, filename)
                            
    #coherenceArr = [[socNet.nodes[ag]['coh'] for ag in agentlist]]
    #nodeCentralityArr = [[socNet.nodes[ag]['nodeCentrality'] for ag in agentlist]]

# %%
