# functions

import numpy as np
import pandas as pd
import networkx as nx
import pingouin
from itertools import combinations
import copy
import string
from scipy.sparse import csr_matrix

def initialise(n, atts, wave, filepath, seed=42):
    # prepare dataset and extract/rename attitude columns
    data = pd.read_csv(filepath, sep=",", index_col="ID", low_memory=False)
    if "gesis" in filepath:
        renameAtts = dict(zip( ["1-kpx_1090", "kpx_1130", "1-kpx_1210", "kpx_1290", "kpx_1250", "kpx_1590"],  ["econ", "migr", "assim", "clim", "euro", "femin"]))
        attOriginalNames = [c for c in data.columns if (len(c.split(" "))==3) and (c.split(" ")[2][1:-1] in renameAtts.keys()) and not "Imp:" in c]
        
        data = data.rename(columns=dict(zip(attOriginalNames, [renameAtts[c.split(" ")[2][1:-1]] for c in attOriginalNames[:]] )))

    if type(wave)==int:
        # single wave
        data = data.loc[data.wave==wave]
    elif type(wave)==list:
        # multiple waves. average per individual (level=0, same index) and take the first entry, which should be the same for all ("the first")
        data = data.loc[data.wave.isin(wave)]
        for att in atts:
            data[att] = data.reset_index().groupby("ID")[att].mean()
        data = data.loc[~data.index.duplicated(keep='first')]
    ops = data[atts].dropna(how="any", axis="index")
    
    if not n=="all":
        ops = ops.sample(n, replace=False, random_state=seed)
    
    agentlist = ops.index.tolist()
    
    # extract identity and weights
    identity = data.loc[agentlist, "partyIdent"]
    if "ess" in filepath:
        weights = data.loc[agentlist, "anweight"]
    else:
        weights = pd.Series(np.ones(len(ops)), index=ops.index)
    return agentlist, weights, ops, identity



hebbian = lambda wij, xi, xj, eps: eps * (1 - np.sign(xi * xj) * wij) * xi *xj

decay = lambda wij, lam: -lam * wij

socialinfluence = lambda wij, Wij, mu: mu * (Wij - wij) if ~np.isnan(Wij) else 0

#dwij = lambda wij, xi, xj, Wij, eps, mu, lam: hebbian(wij, xi, xj, eps) + socialinfluence(wij, Wij, mu) + decay(wij, lam)


def get_socialOmega(agentdict, agentlist, ops, atts, params):
    """if opinions don't change, we can calculate the social signal (Wij) from the correlation or co-occurence upfront

    Args:
        agentdict (dict): contains the attributes (incl opinions, neighbours, ...) of all agents
        agentlist (list): contains indices in the dataframe that define the agents 
        ops (pd.DataFrame): contains opinion values for dimensions (atts) and indices (agentlist)
        atts (list): contains the opinion dimensions
        params (dict): contains parameter settings 

    Returns:
        function: function of ageng ID ag that returns the co-occurence or correlation as seen by agent ag. 
    """
    print("get_socialOmega")
    
    signOp = lambda op: 0 if op==0 else op/abs(op)
    
    if params["socInfType"]=="correlation":
        if params["socNetType"]=="observe-all":
            corrNet = pd.DataFrame( ops.corr()  - np.diag(np.ones(len(atts))),  index=atts, columns=atts)
            return lambda ag: corrNet

    elif params["socInfType"]=="co-occurence":
        if params["socNetType"] == "observe-neighbours": 
            observed_signs = {ag: {att: np.array([signOp(ops.loc[nb, att]) for nb in agentdict[ag]["neighbours"] if nb!= ag]).astype(int) for att in atts} for ag in agentlist}
            W_cooc_ag = {}
            for ag in agentlist:
                W_cooc= [(i, 
                          j,
                          np.nan if len(agentdict[ag]["neighbours"])==0 else 
                          np.mean(observed_signs[ag][i] * observed_signs[ag][j])
                          )
                          for i in atts for j in atts if i != j
                        ]                      
                W_cooc_ag[ag] = pd.DataFrame(W_cooc, columns=["i", "j", "cooccurence"]).pivot_table(index="i", columns="j", values="cooccurence")
            return lambda ag: W_cooc_ag[ag]
        elif params["socNetType"] == "observe-all":
            observed_signs = {att: np.array([signOp(ops.loc[ag, att]) for ag in agentlist]) for att in atts}
            W_cooc = [(i, j, np.mean(observed_signs[i]*observed_signs[j])) for i in atts for j in atts if i != j]
            W_cooc = pd.DataFrame(W_cooc, columns=["i", "j", "cooccurence"]).pivot_table(index="i", columns="j", values="cooccurence")
            return lambda ag: W_cooc
    


#################################
#####  Initialise Network   #####
#################################
def initialise_socialnetwork(agentlist, identity, ops, atts, params ):
    print("initialise_socialnetwork")
    seed = params["seed"]
    np.random.seed(seed)
    if params["socInfType"]=="correlation":
        if params["socNetType"] == "observe-all":
            neighbours = {ag: [] for ag in agentlist}
        else:
            print("ERROR: Correlation can only be run with observe-all")
            quit()
    elif (params["socInfType"]=="co-occurence") or (params["socInfType"]=="copy") :
        if params["socNetType"] == "observe-all":
            neighbours = {ag: [] for ag in agentlist}
        elif params["socNetType"] == "observe-neighbours":            
            groupSizes = [(identity==p).sum() for p in params["parties"]]
            groupSizes[params["parties"].index("none")] += identity.isna().sum() + (identity=="other").sum()
            n = len(agentlist)
            assert (int(sum(groupSizes)) - n) == 0
            A = (np.random.random(size=(n,n))<= params["outdegree"]/n).astype(np.bool_)
            ind = 0
            for group, size in zip(params["parties"], groupSizes):
                # in-group
                A[ind:ind+size, ind:ind+size] = np.random.random(size=(size, size)) <= params["indegree"]/size
                ind += size
            agentlist_sorted = [
                ag for party in params["parties"]
                for ag in agentlist if identity[ag] == party
            ] + [
                ag for ag in agentlist if identity[ag]=="other" or ag in identity.loc[identity.isna()].index
            ]
            #nx.from_numpy_array(A, nodelist=agentlist_sorted)
            agentlist = agentlist_sorted         
            A_csr = csr_matrix(A)
            neighbours = {ag: [agentlist[nb] for nb in A_csr[i].indices.tolist()] for i, ag in enumerate(agentlist)}

    agentdict = {}
    for i, ag in enumerate(agentlist):
        ag_BN = { (a1, a2): 0 for a1, a2 in combinations(atts, 2) }
        attrs =  {
            "x": {att: ops.loc[ag, att] for att in atts} ,
            "identity": identity[ag],
            "BN": ag_BN,
            "coherence": 0,
            "nodeCentrality": [sum([w for e, w in ag_BN.items() if n in e]) for n in atts],
            "neighbours": neighbours[ag]
        }
        agentdict[ag] = attrs

    return agentdict





#################################
#####  coherence    #####
#################################

def coherence(df, atts):
    coherence_edges = [df[i]*df[j]*df[f"({i},{j})"] for i,j in list(combinations(atts, 2))]
    df["coherence"] = np.sum(coherence_edges, axis=0)
    return df 

def coherenceOther(bnEdges, opinion):
    """calculate the coherence of a set of opinions given a personal belief network

    Args:
        bnEdges (pd.Series or dict): a dictionary containing the edge weights of the personal belief network for edges "(a,b)" (keys/index). 
        opinion (dict): a dictionary containing the opinion for which to evaluate the coherence

    Returns:
        float: summed coherence of the opinion given the personal belief network
    """
    atts = opinion.keys() if type(opinion)==dict else (opinion.index if type(opinion)==pd.Series else None)
    coherence_edges = [opinion[i]*opinion[j]*bnEdges[f"({i},{j})"] for i,j in list(combinations(atts, 2))]
    return np.sum(coherence_edges, axis=0)
