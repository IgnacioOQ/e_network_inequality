import json

with open('utils/graph_pipeline/dependency_graph.json') as f:
    g = json.load(f)

nodes = g['nodes']
edges = g['edges']

# Exclude tests and pipeline
targets = [n for n in nodes if n.endswith('.py') and not n.startswith('unit_tests/') and not n.startswith('utils/graph_pipeline/')]

for t in targets:
    deps = [e['source'] for e in edges if e['target'] == t]
    if len(deps) <= 1:
        print(f"{t} is imported by {len(deps)} files: {deps}")
