import ast
import json
import pathlib
import sys
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SKIP = {"__pycache__", ".git", ".venv", "venv", "node_modules", "results"}
DATA_EXTENSIONS = {".csv", ".pkl", ".json", ".txt", ".gml", ".gpickle"}

def collect_files(root):
    py_files = [p for p in root.rglob("*.py") if not any(s in p.parts for s in SKIP)]
    ipynb_files = [p for p in root.rglob("*.ipynb") if not any(s in p.parts for s in SKIP)]
    return py_files, ipynb_files

def extract_notebook_code(ipynb_path):
    try:
        content = json.loads(ipynb_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error parsing notebook {ipynb_path.relative_to(ROOT)}: {e}", file=sys.stderr)
        return ""
    
    code_lines = []
    for cell in content.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, str):
                source = source.splitlines(True)
            for line in source:
                # Strip notebook magics so ast.parse doesn't fail
                if line.strip().startswith("%") or line.strip().startswith("!"):
                    code_lines.append("# " + line)
                else:
                    code_lines.append(line)
            code_lines.append("\n")
    return "".join(code_lines)

def module_map(files, root):
    m = {}
    for f in files:
        parts = list(f.relative_to(root).with_suffix("").parts)
        m[".".join(parts)] = f
    return m

def resolve(name, m):
    if not name:
        return None
    if name in m:
        return m[name]
    best, blen = None, 0
    for k, p in m.items():
        if name.startswith(k + ".") and len(k) > blen:
            best, blen = p, len(k)
    return best

def resolve_relative(name, level, importer_file, m, root):
    importer_dir = importer_file.parent
    package_dir = importer_dir if importer_file.name == "__init__.py" else importer_dir
    
    for _ in range(level - 1):
        package_dir = package_dir.parent
        
    base = ".".join(package_dir.relative_to(root).parts)
    if name:
        full_name = base + "." + name if base else name
    else:
        full_name = base
        
    return resolve(full_name, m)

def edges_for(f, code, m, root):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"SyntaxError parsing {f.relative_to(root)}: {e}", file=sys.stderr)
        return [], []
        
    imports = []
    data_files = []
    
    for n in ast.walk(tree):
        # Imports
        if isinstance(n, ast.Import):
            for a in n.names:
                resolved = resolve(a.name, m)
                if resolved:
                    imports.append(resolved)
        elif isinstance(n, ast.ImportFrom):
            if n.level > 0:
                resolved_mod = resolve_relative(n.module, n.level, f, m, root)
                if resolved_mod:
                    imports.append(resolved_mod)
                for a in n.names:
                    base = n.module if n.module else ""
                    full_name = base + "." + a.name if base else a.name
                    resolved_name = resolve_relative(full_name, n.level, f, m, root)
                    if resolved_name:
                        imports.append(resolved_name)
            else:
                if n.module:
                    resolved = resolve(n.module, m)
                    if resolved:
                        imports.append(resolved)
                    for a in n.names:
                        full_name = n.module + "." + a.name
                        resolved_name = resolve(full_name, m)
                        if resolved_name:
                            imports.append(resolved_name)
        
        # Data Files (string literals)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            val = n.value
            if any(val.endswith(ext) for ext in DATA_EXTENSIONS):
                # Clean up path if it has leading ./ or /
                val = val.strip()
                if val.startswith("./"):
                    val = val[2:]
                elif val.startswith("/"):
                    val = val[1:]
                data_files.append(val)
                
    return [p for p in imports if p and p != f], data_files

def build():
    py_files, ipynb_files = collect_files(ROOT)
    m = module_map(py_files, ROOT)
    
    nodes = set()
    edges = []
    
    # Process Python files
    for f in py_files:
        code = f.read_text(encoding="utf-8")
        imports, data = edges_for(f, code, m, ROOT)
        
        f_rel = str(f.relative_to(ROOT))
        nodes.add(f_rel)
        
        for t in imports:
            t_rel = str(t.relative_to(ROOT))
            nodes.add(t_rel)
            edges.append({"source": f_rel, "target": t_rel, "type": "import"})
            
        for d in data:
            nodes.add(d)
            edges.append({"source": f_rel, "target": d, "type": "data_io"})
            
    # Process Notebooks
    for f in ipynb_files:
        code = extract_notebook_code(f)
        imports, data = edges_for(f, code, m, ROOT)
        
        f_rel = str(f.relative_to(ROOT))
        nodes.add(f_rel)
        
        for t in imports:
            t_rel = str(t.relative_to(ROOT))
            nodes.add(t_rel)
            edges.append({"source": f_rel, "target": t_rel, "type": "import"})
            
        for d in data:
            nodes.add(d)
            edges.append({"source": f_rel, "target": d, "type": "data_io"})
            
    # deduplicate edges
    unique_edges = []
    seen = set()
    for e in edges:
        tup = (e["source"], e["target"], e["type"])
        if tup not in seen:
            seen.add(tup)
            unique_edges.append(e)
            
    return {"nodes": sorted(list(nodes)), "edges": unique_edges}

if __name__ == "__main__":
    result = build()
    out_file = ROOT / "utils" / "graph_pipeline" / "dependency_graph.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Graph written to {out_file.relative_to(ROOT)}")
    print(f"Nodes: {len(result['nodes'])}")
    print(f"Edges: {len(result['edges'])}")

