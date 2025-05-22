#%%
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
from model_dynamic import *
from analysis_help import *
import sys

sys.path.append("code/")
smallfs = 10
bigfs = 13
supersmallfs =9
plt.rcParams.update({"font.size":bigfs})

NetworkTypes = ["observe-all"]
InfluenceTypes = ["copy", "co-occurence"]
n_ag = 100
M = 4
seed = 1
indegree= 10
outdegree = 0
ngroups = 0
belief_jump = 0.1
resultsfolder = "results-dynNoB/"

#%%
#################################
#####  Plot the time dynamics   #####
#################################

smallfs = 10
bigfs = 13
supersmallfs =9
plt.rcParams.update({"font.size":bigfs})

partyColsDict = {"none":"red"}
atts = list(string.ascii_lowercase[:M])
edgeNamesTuple = [f"({i},{j})" for i,j in list(combinations(atts, 2))]
edgeNames = [f"{e.split(',')[0][1:]}↔{e.split(",")[1][:-1]}" for e in edgeNamesTuple]
for mu in [0.4]:
    eps = 0.1
    lam = 0.0
    for T in [0.01]:
        eps, mu, lam, Temp, inf, net = (eps,mu,lam, T, "copy", "observe-neighbours")
        initial_w = 0.1  if (eps,mu,lam)==(0.,0.,0.) else 0.0

        fname = resultsfolder+f"dynamicNoB_M-{M}_n-{n_ag}_{inf}-{net}"+(f"(Stoch-{ngroups}-Block-{indegree}-{outdegree})" if "neighbours" in net else "")+f"_beliefJump-{belief_jump}-T{Temp}"+f"_eps{eps}_mu{mu}_lam{lam}_initialW-{initial_w}_seed{seed}"

        sim = pd.read_csv(fname+".csv")
        sims = pd.read_csv(f"{fname}_overTime.csv")

        fig, axs = plt.subplot_mosaic([["a", "a2", "x","b", "b2"], ["ab","ab","ab", "ab", "ab2"]], sharey=True, gridspec_kw={"width_ratios":[1,0.2,0.05,1,0.2]})
        axs["x"].axis("off")
        i, j = ("a", "b") 
        e = f"({i},{j})" if f"({i},{j})" in edgeNamesTuple else f"({j},{i})"
        for ax, feature in zip(["a", "b", "ab"], [i, j, e]) :
            a = plot_results_over_time(sims,  feature, partyColsDict, axs=[axs[ax], axs[ax+"2"]], n_sample=100)
            axs[ax].set_xlabel("")
        axs["ab"].set_xlabel("time t")
        axs["ab"].set_ylabel("edge weight")
        axs["a"].set_ylabel("belief score")
        fig.suptitle(fr"$\epsilon={eps}$, $\mu={mu}$, $\lambda={lam}$, Temp={T},"+"\n"+rf"{inf}, {net}, seed {seed}", fontsize=bigfs)    
        fig.tight_layout(w_pad=0)



#%%
#################################
#####  PLOT DISTRIBUTIONS    #####
#################################

# smallfs =9 
# bigfs = 7
# supersmallfs =6
# plt.rcParams.update({"font.size":bigfs})

# seed = 1
# epsarr = [0.05,0.4, 0.8]
# muarr = [0.05,0.2, 0.4]
# T = 0.01
# fig, axs = compare(0.4, 0.2,0.05,T,"copy", "observe-neighbours", epsarr, muarr, seed, M, n_ag, ngroups, indegree, outdegree, belief_jump, resultsfolder,  edgeNamesTuple, atts)
