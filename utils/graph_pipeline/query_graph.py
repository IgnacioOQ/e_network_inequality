import json
with open('utils/graph_pipeline/dependency_graph.json') as f:
    g = json.load(f)

edges = g['edges']
nodes = g['nodes']

# helper: who imports/uses X?
def dependents_of(target):
    return [e['source'] for e in edges if e['target'] == target]

# 1. networks/network_generation.py
print(f"network_generation.py is imported by: {dependents_of('networks/network_generation.py')}")

# 3. tobacco_extended_network.pkl
print(f"tobacco_extended_network.pkl is used by: {dependents_of('networks/citation_data/tobacco_extended_network.pkl')}")
print(f"tobacco_extended_works.pkl is used by: {dependents_of('networks/citation_data/tobacco_extended_works.pkl')}")

# 4. utils scripts
for u in [n for n in nodes if n.startswith('utils/') and n.endswith('.py')]:
    deps = dependents_of(u)
    print(f"{u} is used by: {deps}")
