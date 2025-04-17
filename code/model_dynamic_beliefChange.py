
#%% 

import numpy as np
import pandas as pd
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
            if len(agentdict[ag]['neighbours']) == 0 and not params["socNetType"]=="observe-all":
                currWij = 0
            else:
                neighbours = agentdict[ag]['neighbours'] if params["socNetType"]=="observe-neighbours" else list(agentdict.keys())
                sampled_neighbour = np.random.choice(neighbours)
                currWij = agentdict[sampled_neighbour]['BN'][e]
        else:
            currWij = 0 if Wij(ag).empty else Wij(ag).loc[i,j]
        delBeta += socialinfluence(wij, currWij, mu)  

        BN_ag[e] = np.clip(wij + params["dt"] * delBeta, a_min=-1, a_max=1)

    # node updating. 
    # TODO: randomise? shuffle
    for att, b in x.items():
        # options are b-0.1, b, b+0.1
        options = [max(-1, b-params["belief_jump"]), b, min(1, b+params["belief_jump"])]
        ps = glauber_probabilities(att, options, x, BN_ag, params["Temp"], atts)
        x[att] = np.random.choice(options, p=ps)

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

def dynSim(filepath, atts, wave, params, predefined_agentlist):
    np.random.seed(params["seed"])
    
    agentlist, weights, ops, identity = initialise(filepath, wave, params["n"], predefined_agentlist, atts, params["seed"])
    print(f"Nr of agents: {len(agentlist)}: " , identity.value_counts())


    agentdict = initialise_socialnetwork(agentlist, identity, ops, atts, params)
    agentlist = list(agentdict.keys())
    agentlistOrig = np.copy(agentlist)

    if (params["socInfType"]=="co-occurence") or (params["socInfType"]=="correlation"):
        Wij = get_socialOmega(agentdict, agentlist, ops, atts, params)
    else:
        Wij = None

    print("simulate", end="...")
    time = np.arange(0,params["T"], step=params["dt"])
    results_over_time = []
    
    # simulate time steps
    for t in time[1:]:
        if (t%10==0): 
            print(t, end=", ")
        np.random.shuffle(agentlist)
        for ag in agentlist:
            agentdict = update_step(t, ag, agentdict, atts, Wij, params)
        if t in params["track_times"]:
            results_over_time.append([[agentdict[ag]['BN'][e] for e in agentdict[ag]['BN'].keys()] + [agentdict[ag]['x'][att] for att in atts] for ag in agentlistOrig])
        #    coherenceArr.append([agentdict[ag]['coherence'] for ag in agentlist])
        #    nodeCentralityArr.append([agentdict[ag]['nodeCentrality'] for ag in agentlist])
    print("done")

    edgelist = [f"({e[0]},{e[1]})" for e in agentdict[agentlistOrig[0]]['BN'].keys()]
    simOut = pd.DataFrame(
        data=[[agentdict[ag]['BN'][e] for e in agentdict[ag]['BN'].keys()] + [agentdict[ag]['x'][att] for att in atts] for ag in agentlistOrig], # x and w's (columns) per agent (row)
        columns = edgelist + atts, index=agentlistOrig)
    simOut.loc[agentlistOrig, "agent_weight"] = weights[agentlistOrig]
    simOut.loc[agentlistOrig, "identity"] = identity[agentlistOrig]
    simOut = simOut.reset_index()
    return agentdict, simOut, results_over_time, agentlistOrig


# %%
#################################
#####  MAIN   #####
#################################
if __name__=="__main__":
    folder = ""
    inputfolder = folder+"inputdata/" #"~/csh-research/projects/opinion-data-curation/data/clean/"
    dataset = "liss"
    for dataset in ["liss", "gesis"]:
        country= countryXdataset[dataset]
        atts = atts_datasets[dataset]
        edgeNames = [f"({i},{j})" for i,j in list(combinations(atts, 2))]
        params={
            "n": 1000,    # integer or "all"
            "seed":5,
            "T":100,
            "dt":1,
            "track_times": np.arange(0,100, 1),
            "socNetType":"observe-neighbours",  # observe-neighbours or observe-all
            "socInfType":"copy",   # correlation or co-occurence or copy
            "eps":None, #filled later
            "mu":None, 
            "lam":None,
            "parties": parties[country], 
            "indegree":10, 
            "outdegree":0,
            "belief_jump":0.1,
            "Temp":1,
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
        
        filepath = f"{inputfolder}{dataset}-{country.lower()}.csv"
        
        predefined_agentlist = None if params["n"]=="all" else determine_agentlist(filepath, waves[dataset], params["n"], atts, params["seed"])
        for eps, mu, lam in paramCombis:
            params["eps"] = eps
            params["mu"]  = mu
            params["lam"] = lam
            

            for wave in waves[dataset]:
                agentdict, simOut, results_over_time, agentlistOrig = dynSim(filepath, atts, wave, params, predefined_agentlist)
                if predefined_agentlist is not None:
                    agentdict = {ag:agentdict[ag] for ag in predefined_agentlist}
                waveName = wave if type(wave)==int else f"{wave[0]}-{wave[-1]}"
                filename = folder+f"results/inferBNs+BeliefChange-dynamic_{dataset.lower()}-n-{params['n']}_{params['socInfType']}-{params['socNetType']}"+(f"(Stoch-Block-{params['indegree']}-{params['outdegree']})" if "neighbours" in params["socNetType"] else "")+f"beliefJump-{params['belief_jump']}-T{params['Temp']}"+f"_{waveName}-{country}_eps{eps}_mu{mu}_lam{lam}_seed{params['seed']}"
                simOut.to_csv(filename+".csv")

                print(eps,mu,lam, filename)

                # Store results over time
                edgelist = list(simOut.columns[1:-8])
                sims = []
                for t, res in zip(params["track_times"] , results_over_time):
                    sim = pd.DataFrame(
                        data=res, # x and w's (columns) per agent (row)
                        columns = edgelist + atts, index=agentlistOrig)
                    sim.loc[agentlistOrig, "agent_weight"] = simOut["agent_weight"]
                    sim.loc[agentlistOrig, "identity"] = simOut["identity"]
                    sim["t"] = t
                    sim = sim.reset_index()
                    sim = sim.rename(columns={"index":"ID"})
                    sims.append(sim)
                sims = pd.concat(sims)
                sims.to_csv(filename+"_overTime.csv")


#%%

