"""ARCHIVED — historical one-shot migration script. Do not run.

Written for a repository layout that no longer exists: it targets
`e_network_inequality/notebooks/network_variations/` and rewrites paths to point at
`../../src/net_epistemology/utils` and `../../data/empirical_networks/`. None of those
directories are present in the current tree — the notebook it patches now lives at
`testing/notebooks/variation_methods_test.ipynb`, and the networks at
`networks/citation_data/`.

Kept only as a record of the path-migration that produced the current layout.
"""

import json
import os

notebook_path = 'e_network_inequality/notebooks/network_variations/variation_methods_test.ipynb'

try:
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    # Modifying imports
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            # Join line list to check content easily, or iterate
            # Check if this is the import cell
            if any('import dill' in line for line in source) and not any('sys.path.append' in line for line in source):
                # Find index of "import dill"
                dill_idx = -1
                for i, line in enumerate(source):
                    if 'import dill' in line:
                        dill_idx = i
                        break
                
                if dill_idx != -1:
                    # Insert sys.path before import dill
                    new_lines = [
                        "import sys\n",
                        "import os\n",
                        "\n",
                        "# Add utils directory to path to allow imports of variation_methods and network_utils\n",
                        "# and to let them find their local dependencies (imports.py)\n",
                        "sys.path.append(os.path.abspath(os.path.join(os.getcwd(), \"../../src/net_epistemology/utils\")))\n",
                        "\n"
                    ]
                    # Insert before dill
                    # Use slicing to insert multiple items
                    cell['source'] = source[:dill_idx] + new_lines + source[dill_idx:]
                    print("Updated imports.")

    # Modifying data path
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            for i, line in enumerate(source):
                if "'empirical_networks/pud_final.pkl'" in line:
                    source[i] = line.replace("'empirical_networks/pud_final.pkl'", "'../../data/empirical_networks/pud_final.pkl'")
                    print("Updated data path.")

    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print("Notebook updated successfully.")

except Exception as e:
    print(f"Error: {e}")
