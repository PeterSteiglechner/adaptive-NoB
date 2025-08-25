import networkx as nx 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np

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
