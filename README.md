# Adaptive version of Belief Network Theory

We take opinions from Netherlands/Germnay/Austria(TODO) panel data in two wave periods (before/after 2021). We calculate how indivduals with these opinions form belief networks based on three processes:
1. Hebbian Learning: $\Delta w_{ij} += \epsilon (1- sign(x_i \cdot x_j) \cdot w_{ij}) \cdot x_i  \cdot x_j$
2. Decay:  $\Delta w_{ij} += - \lambda w_{ij}$
3. Social influence: $\Delta w_{ij} += \mu \cdot (\Omega_{ij}  - w_{ij})$, where $\Omega_{ij}$ is the
* Correlation (of the opinions of all others) between topic i and j
* Co-occurence of aligned/misaligned opinions on topics i and j (either across society or among network neighbours).
* a copy of the belief network edge between issues i and j of some neighbouring agent (or the mean of that). NOT YET FULLY IMPLEMENTED    


There are two versions of the model: 

1. model_equ.py assumes that beliefs remain constant and individuals observe the correlation/co-occurence of beliefs of **all** others. Thus, we can define an equilibrium model in which $\Delta w_{ij}=0$ for all belief dimensions $i$ and $j$.

2. model_dynamic.py can incorporate also belief change or networks, such that agent observations are heterogeneous 

## Run
1. unzip inputdata.zip
2. create a folder "results"
3. in the ``` __name__=='__main__' ``` adjust the followign as you wish:
- dataset = "gesis"  (Germany Panel data) or "liss" (Netherlands Panel data)
- in params:
     - "n": this can be either an integer, so that we sample randomly from the dataset n individuals. Or you can set it to "all" so that we use all valid participants (which provided at least one answer for the surveys of the specified waves) 
     - "socNetType":
          - "observe-all" --> agents observe all others, we don't need to bother about networks
          - "observe-neighbours" --> agents observe only network neighhbours. This only makes sense for co-occurence (and later copying)
     - "socInfType":
          - "co-occurence",
          - "correlation" (only if socNetType is observe-all)
          - "copy" (NOT YET)
- in paramCombis sepcify a list of simulation runs with  ``` [(eps1, mu1, lam1), (eps2, mu2, lam2), ...] ```
- Note that params "parties", "indegree", "outdegree" only change the social network, which is currently conceptualised as a stochastic block network based on parties that individuals feel closest to. We aim to ensure that each agent has on average indegree + outdegree links.     
    }
