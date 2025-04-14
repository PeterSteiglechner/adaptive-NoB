
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

def update_step(t, ag, agentdict, atts, Wij, params, **kwargs):
    eps, mu, lam = (params["eps"], params["mu"], params["lam"])
    x = agentdict[ag]['x']
    BN_ag = agentdict[ag]['BN']
    
    #  EDGES - Hebbian + Social + Decay
    edgelist = list(BN_ag.keys())
    for n_edge in np.random.choice(range(len(edgelist)), size=len(edgelist), replace=False):      
        e = edgelist[n_edge]
        i,j = e
        wij = BN_ag[e]
        # Decay
        delBeta = decay(wij, lam)
        # Hebbian
        delBeta += hebbian(wij, x[i], x[j], eps)
        # Social
        if params["socInfType"]=="copy":
            if len(agentdict[ag]['neighbours']) == 0:
                currWij = 0
            else:
                neighbour = np.random.choice(agentdict[ag]['neighbours'])
                currWij = agentdict[neighbour]['BN'][e]
        else:
            currWij = 0 if Wij(ag).empty else Wij(ag).loc[i,j]
        delBeta += socialinfluence(wij, currWij, mu)  

        BN_ag[e] = np.clip(wij + params["dt"] * delBeta, a_min=-1, a_max=1)

    # TODO insert here node updating. 

    # 

    # SUMMARISE
    agentdict[ag]['BN'] = BN_ag
    agentdict[ag]['x'] = x

    if t in params["track_times"]:
        edge_weights = np.array([BN_ag[e] for e in edgelist])
        coherence_values = edge_weights * np.array([x[e[0]] * x[e[1]] for e in edgelist])    
        coherence_dict = {n: coherence_values[np.array([n in e for e in edgelist])].sum() for n in atts}
        coherence_node = [coherence_dict[n] for n in atts]
        agentdict[ag]["coherence"] =  np.sum(coherence_node)
        agentdict[ag]["nodeCentrality"] = [sum(abs(BN_ag[e]) for e in edgelist if n in e) for n in atts]
    return agentdict

    
#%% 
#################################
#####  Simulation Run   #####
#################################

def dynSim(atts, wave, country, params, folder, dataset):
    np.random.seed(params["seed"])
    
    filepath = f"{folder}{dataset}-{country.lower()}.csv"
    print(filepath)
    agentlist, weights, ops, identity = initialise(params["n"], atts, wave, filepath)
    print(f"Nr of agents: {len(agentlist)}: " , identity.value_counts())


    agentdict = initialise_socialnetwork(agentlist, identity, ops, atts, params)
    agentlist = list(agentdict.keys())

    if (params["socInfType"]=="co-occurence") or (params["socInfType"]=="correlation"):
        Wij = get_socialOmega(agentdict, agentlist, ops, atts, params)
    else:
        Wij = None

    print("simulate", end="...")
    time = np.arange(0,params["T"], step=params["dt"])
    for t in time[1:]:
        if (t%10==0): 
            print(t, end=", ")
        np.random.shuffle(agentlist)
        for ag in agentlist:
            agentdict = update_step(t, ag, agentdict, atts, Wij, params)
        #if t in params["track_times"]:
        #    coherenceArr.append([agentdict[ag]['coherence'] for ag in agentlist])
        #    nodeCentralityArr.append([agentdict[ag]['nodeCentrality'] for ag in agentlist])
    print("done")

    edgelist = [f"({e[0]},{e[1]})" for e in agentdict[agentlist[0]]['BN'].keys()]
    simOut = pd.DataFrame(
        data=[[agentdict[ag]['BN'][e] for e in agentdict[ag]['BN'].keys()] + list(ops.loc[ag, atts]) for ag in agentlist], # x and w's (columns) per agent (row)
        columns = edgelist + atts, index=agentlist)
    simOut.loc[agentlist, "agent_weight"] = weights[agentlist]
    simOut.loc[agentlist, "identity"] = identity[agentlist]
    simOut = simOut.reset_index()
    return agentdict, simOut


# %%
#################################
#####  MAIN   #####
#################################
if __name__=="__main__":
    folder = "inputdata/" #"~/csh-research/projects/opinion-data-curation/data/clean/"
    dataset = "gesis"
    country= countryXdataset[dataset]
    atts = atts_datasets[dataset]
    edgeNames = [f"({i},{j})" for i,j in list(combinations(atts, 2))]
    params={
        "n": 1000,    # integer or "all"
        "seed":42,
        "T":100,
        "dt":1,
        "track_times":[0,100],
        "socNetType":"observe-neighbours",  # observe-neighbours or observe-all
        "socInfType":"copy",   # correlation or co-occurence or copy(TODO)
        "eps":None, #filled later
        "mu":None, 
        "lam":None,
        "parties": parties[country], 
        "indegree":10, 
        "outdegree":0,
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
            agentdict, simOut = dynSim(atts, wave, country, params, folder=folder, dataset=dataset)
            waveName = wave if type(wave)==int else f"{wave[0]}-{wave[-1]}"
            filename = f"results/inferBNs-dynamic_{dataset.lower()}-n-{params['n']}_{params['socInfType']}-{params['socNetType']}"+(f"(Stoch-Block-{params['indegree']}-{params['outdegree']})" if "neighbours" in params["socNetType"] else "")+f"_{waveName}-{country}_eps{eps}_mu{mu}_lam{lam}_seed{params['seed']}.csv"
            simOut.to_csv(filename)

            print(eps,mu,lam, filename)
                            
    #coherenceArr = [[agentdict[ag]['coherence'] for ag in agentlist]]
    #nodeCentralityArr = [[agentdict[ag]['nodeCentrality'] for ag in agentlist]]

# %%
