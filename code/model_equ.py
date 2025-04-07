import numpy as np
import pandas as pd
import networkx as nx
import pingouin
from itertools import combinations
import copy
import string
from help_functions  import *

countryXdataset = {"gesis":"Germany", "liss":"Netherlands", "autnes":"Austria"}
atts_datasets = {
    "gesis": ["econ", "migr", "assim", "clim", "euro", "femin"], #["1-kpx_1090", "kpx_1130", "1-kpx_1210", "kpx_1290", "kpx_1250", "kpx_1590"],
    "liss": ["euth", "inequ", "migrAss", "eu", "nr_mig", "asyl"],
    "autnes":[]
}
parties = {
    "Germany": ["Linke", "Grüne", "SPD", "FDP", "CDU/CSU", "AfD", "none"],
    "Netherlands": ["far left", "centre left", "centre", "centre right", "far right", "none"],
    "Austria": ["Grüne", "SPÖ", "NEOS", "ÖVP", "FPÖ", "none"]
}
#################################
#####  Equilibrium Model    #####
#################################

def equilibrium_wij(xi, xj, Wij, eps, mu, lam):
    if mu==0 and lam==0 and (xi*xj)==0:
        return 0
    return (mu * Wij + eps * xi * xj) / (eps * np.abs(xi * xj) + mu + lam)   
# note: sign(xi xj) xi xj == abs(xi xj)

def equiSim(atts, wave, params, country, folder, dataset):
    assert (params["socNetType"]=="observe-all")
    eps, mu, lam = (params["eps"], params["mu"], params["lam"])
    edgeNames = [f"({i},{j})" for i,j in list(combinations(atts, 2))]
    filepath = f"{folder}{dataset}-{country.lower()}.csv"
    print(filepath)
    agentlist, weights, ops, identity = initialise(n="all", atts=atts, wave=wave, filepath=filepath)
    # calculate correlation network
    Wij = get_socialOmega(agentlist, ops, dict(zip(agentlist, agentlist)), atts, params)
    # calculate equilibrium edges
    bnlist = []
    for ag in agentlist:
        op = ops.loc[ag]
        edgeWeights = [equilibrium_wij(op[i], op[j], Wij(ag).loc[i,j], eps, mu,lam) for i,j in combinations(atts, 2)]
        bn_ag = [ag, weights[ag], identity[ag]] +  list(op[atts].values) + edgeWeights
        bnlist.append(bn_ag)
    return pd.DataFrame(bnlist, columns=["agent", "weight", "identity"] + atts + edgeNames)


#%%
#################################
#####  MAIN   #####
#################################


if __name__=="__main__":
    folder = "~/csh-research/projects/opinion-data-curation/data/clean/"
    dataset = "gesis"
    country= countryXdataset[dataset]
    atts = atts_datasets[dataset]
    edgeNames = [f"({i},{j})" for i,j in list(combinations(atts, 2))]
    params={
        "socNetType":"observe-all", # no other option
        "socInfType":"co-occurence", # correlation or co-occurence
        "eps":None, #filled later
        "mu":None, 
        "lam":None,
        "parties": parties[country],
    }

    waves = {
        "gesis":[list(range(1,20)), list(range(20,28))],
        "liss":[
            [string.ascii_lowercase[y-8-1] for y in [16,17,18,19,20]],  #before
            [string.ascii_lowercase[y-8-1] for y in [21,22,23,24]] #after
            ]
    }
    paramCombis = [(0.0,0.2,0.05), (0.4,0.0,0.05), (0.4,0.2,0.05)]
    for eps, mu, lam in paramCombis:
        params["eps"] = eps
        params["mu"]  = mu
        params["lam"] = lam
        for wave in waves[dataset]:
            simOut = equiSim(atts=atts, wave=wave, params=params, country=country, folder=folder, dataset=dataset)
            waveName = wave if type(wave)==int else f"{wave[0]}-{wave[-1]}"
            filename = f"resultsMar/inferBNs-equilibrium_{params['socInfType']}_{dataset.upper()}{waveName}-{country}_eps{eps}_mu{mu}_lam{lam}.csv"
            simOut.to_csv(filename)
            print(eps,mu,lam, filename)
                            


# %%
