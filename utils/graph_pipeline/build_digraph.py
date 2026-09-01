"""
Pipeline script to build a dependency digraph of the codebase.
This script statically parses Python files and Jupyter notebooks using the `ast` module
to extract import dependencies and data file I/O operations without running the code.
"""

import ast
import json
import pathlib
import sys

# The root directory of the repository, used to resolve absolute and relative paths.
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# Directories to ignore during traversal to prevent noise in the graph.
SKIP = {"__pycache__", ".git", ".venv", "venv", "node_modules", "results", "convergence_analysis", "graph_pipeline"}

# File extensions that indicate data I/O operations. 
# Used by the AST visitor to track which scripts read/write which data artifacts.
DATA_EXTENSIONS = {".csv", ".pkl", ".json", ".txt", ".gml", ".gpickle"}

def collect_files(root):
    """Recursively collects all Python (.py) and Jupyter Notebook (.ipynb) files, skipping ignored directories."""
    py_files = [p for p in root.rglob("*.py") if not any(s in p.parts for s in SKIP)]
    ipynb_files = [p for p in root.rglob("*.ipynb") if not any(s in p.parts for s in SKIP)]
    return py_files, ipynb_files

def extract_notebook_code(ipynb_path):
    """
    Parses a Jupyter Notebook's JSON structure and extracts all Python code cells into a single string.
    Jupyter magic commands (like %time or !pip) are commented out so they don't break the standard Python AST parser.
    """
    try:
        content = json.loads(ipynb_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error parsing notebook {ipynb_path.relative_to(ROOT)}: {e}", file=sys.stderr)
        return ""
    
    code_lines = []
    for cell in content.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            # Some Jupyter formats store source as a single string instead of a list of lines
            if isinstance(source, str):
                source = source.splitlines(True)
            for line in source:
                # Strip notebook magics so ast.parse doesn't fail with a SyntaxError
                if line.strip().startswith("%") or line.strip().startswith("!"):
                    code_lines.append("# " + line)
                else:
                    code_lines.append(line)
            code_lines.append("\n")
    return "".join(code_lines)

def module_map(files, root):
    """
    Creates a lookup registry mapping fully-dotted module names (e.g., 'utils.network_utils')
    to their absolute file paths (e.g., ROOT/utils/network_utils.py).
    This allows us to resolve 'import utils.network_utils' back to a specific file.
    """
    m = {}
    for f in files:
        parts = list(f.relative_to(root).with_suffix("").parts)
        m[".".join(parts)] = f
    return m

def resolve(name, m):
    """
    Attempts to resolve a dotted import name against the module registry.
    It uses longest-prefix matching to handle imports like 'from model.agents import BetaAgent',
    where 'model.agents' is the file and 'BetaAgent' is the class.
    """
    if not name:
        return None
    if name in m:
        return m[name]
    
    # Fallback: Find the longest matching prefix (e.g. matching 'model.agents' from 'model.agents.BetaAgent')
    best, blen = None, 0
    for k, p in m.items():
        if name.startswith(k + ".") and len(k) > blen:
            best, blen = p, len(k)
    return best

def resolve_relative(name, level, importer_file, m, root):
    """
    Resolves relative imports (e.g., 'from . import utils' or 'from ..model import agents').
    'level' corresponds to the number of dots (1 = current directory, 2 = parent, etc.).
    """
    importer_dir = importer_file.parent
    # __init__.py files act as the directory itself for relative imports
    package_dir = importer_dir if importer_file.name == "__init__.py" else importer_dir
    
    # Traverse up the directory tree for each dot beyond the first
    for _ in range(level - 1):
        package_dir = package_dir.parent
        
    base = ".".join(package_dir.relative_to(root).parts)
    if name:
        full_name = base + "." + name if base else name
    else:
        full_name = base
        
    return resolve(full_name, m)

def edges_for(f, code, m, root):
    """
    Walks the Abstract Syntax Tree (AST) of the provided code to extract dependencies.
    Returns a tuple of two lists:
    1. Local Python file imports (resolved paths).
    2. Data files accessed (string literals ending in known data extensions).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"SyntaxError parsing {f.relative_to(root)}: {e}", file=sys.stderr)
        return [], []
        
    imports = []
    data_files = []
    
    for n in ast.walk(tree):
        # Handle 'import X'
        if isinstance(n, ast.Import):
            for a in n.names:
                resolved = resolve(a.name, m)
                if resolved:
                    imports.append(resolved)
                    
        # Handle 'from X import Y'
        elif isinstance(n, ast.ImportFrom):
            if n.level > 0:
                # Relative import (e.g., from .module import X)
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
                # Absolute import (e.g., from utils.network_utils import X)
                if n.module:
                    resolved = resolve(n.module, m)
                    if resolved:
                        imports.append(resolved)
                    for a in n.names:
                        full_name = n.module + "." + a.name
                        resolved_name = resolve(full_name, m)
                        if resolved_name:
                            imports.append(resolved_name)
        
        # Handle Data Files (string literals like pd.read_csv("data.csv"))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            val = n.value
            if any(val.endswith(ext) for ext in DATA_EXTENSIONS):
                # Clean up path if it has leading ./ or / for consistency
                val = val.strip()
                if val.startswith("./"):
                    val = val[2:]
                elif val.startswith("/"):
                    val = val[1:]
                data_files.append(val)
                
    # Filter out self-imports and Nones
    return [p for p in imports if p and p != f], data_files

def build():
    """
    Main pipeline orchestrator:
    1. Collects all scripts and notebooks.
    2. Extracts their edges (imports + data I/O).
    3. Compiles a deduplicated list of nodes and edges.
    """
    py_files, ipynb_files = collect_files(ROOT)
    m = module_map(py_files, ROOT)
    
    nodes = set()
    edges = []
    
    # Process standard Python modules
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
            
    # Process Jupyter Notebooks
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
            
    # Deduplicate edges (in case a file imports the same module twice)
    unique_edges = []
    seen = set()
    for e in edges:
        tup = (e["source"], e["target"], e["type"])
        if tup not in seen:
            seen.add(tup)
            unique_edges.append(e)
            
    return {"nodes": sorted(list(nodes)), "edges": unique_edges}

if __name__ == "__main__":
    # Execute the pipeline and write the resulting graph to JSON
    result = build()
    out_file = ROOT / "utils" / "graph_pipeline" / "dependency_graph.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Graph written to {out_file.relative_to(ROOT)}")
    print(f"Nodes: {len(result['nodes'])}")
    print(f"Edges: {len(result['edges'])}")

