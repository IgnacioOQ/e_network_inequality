# Markdown-METADATA Hybrid Schema Conventions
- status: active

This document defines the strict conventions for the Markdown-METADATA Hybrid Schema used in this project for hierarchical task coordination and agentic planning.

## Core Principle
- status: active

The system uses a **Markdown headers** to define the structural hierarchy (the nodes) and **METADATA Frontmatter-style blocks** (immediately following the header) to define structured metadata.

## Schema Rules
- status: active

### 1. Hierarchy & Nodes
- status: active

- **Headers**: Use standard Markdown headers (`#`, `##`, `###`) to define the hierarchy.
- **Nesting**: 
    - `#` is the Root/Document Title (usually only one per file).
    - `##` are Top-Level Tasks/Features.
    - `###` are Subtasks.
    - And so on.
- **Implicit Dependency**: A header is implicitly dependent on its parent header.
- **Explicit Dependency (DAG)**: 
    - Use the `blocked_by` metadata field to define dependencies on other nodes (siblings, cousins, etc.).
    - Pass a list of IDs or relative paths to allow a single node to depend on multiple prior nodes.
    - **Example**: `blocked_by: [task-a, task-b]` implies this node cannot start until both 'task-a' and 'task-b' are done.

### 2. Metadata Blocks
- status: active

- **Location**: Metadata MUST be placed **immediately** after the header.
- **Separator**: There MUST be a blank line between the metadata block and the content.
- **Format**: A METADATA block. It works best as a bulleted list of key-value pairs.
- **Preferred Format**: A strict list of key-value pairs.

**Example:**

```markdown
## Implement User Auth
- status: in-progress
- type: episodic
- owner: dev-1
- estimate: 3d
- blocked_by: []

```

### 3. Allowed Fields
- status: active

The following fields are standard, but the schema allows extensibility.

| Field | Type | Description |
| :--- | :--- | :--- |
| `status` | `enum` | `todo`, `in-progress`, `done`, `blocked` |
| `type` | `enum` | `recurring`, `episodic`, `binary` |
| `owner` | `string` | The agent or user assigned to this (e.g., `dev-1`, `claude`) |
| `estimate` | `string` | Time estimate (e.g., `1d`, `4h`) |
| `blocked_by`| `list` | List of explicit dependencies (IDs or relative paths) |
| `priority` | `enum` | `low`, `medium`, `high`, `critical` (Optional) |
| `id` | `string` | Unique identifier for the node (e.g., `project.component.task`). Used for robust merging and dependency tracking. |
| `last_checked` | `string` | This is the date of the last time this node was modified, including change of status. |

For extended fields consider:
 - The key is entirely lowercase
 - The key has no spaces (words are separated with dash or underscore)
 - The value is single line

### 4. Context & Description
- status: active

- Any text following the metadata block is considered "Context" or "Description".
- It can contain free-form Markdown, code blocks, images, etc.

## Examples
- status: active

### Valid Node
- status: active

```markdown
### Database Schema
- status: done
- owner: dev-2
- estimate: 1d

Set up PostgreSQL schema for users and sessions.
```

### Invalid Node (Metadata not immediate)
- status: active

```markdown
### Database Schema
- status: active

Some text here first.

- status: done

```
*Error: Metadata block must immediately follow the header.*

### Invalid Node (Bad indentation/METADATA)
- status: active

```markdown
### Database Schema
status: done
owner: dev-2

```
*Warning: While some parsers might handle this, prefer bullet points `- key: value` for readability and stricter parsing.*

## Parsing Logic (for Developers)
- status: active

1. **Scan for Headers**.
2. **Look ahead** at the lines immediately following the header.
3. **Parse lines** that match the METADATA key-value pattern (`- key: value` or `key: value`) until a blank line or non-matching line is found.
4. **Everything else** until the next header of equal or higher level is "Content".

## Tooling Reference
- status: active

The following Python scripts are available in `language/` to interact with this schema:

### 1. `language/md_parser.py`
- status: active

- **Purpose**: Parses `.md` files into a Python object tree and validates schema compliance.
- **Usage**: `python3 language/md_parser.py <file.md>`
- **Output**: JSON representation of the tree or validation errors.

### 2. `language/visualization.py`
- status: active

- **Purpose**: Visualizes the task tree in the terminal with metadata.
- **Usage**: `python3 language/visualization.py <file.md>`
- **Output**: Unicode tree visualization.

### 3. `language/operations.py`
- status: active

- **Purpose**: Manipulate task trees (merge, extend).
- **Usage**:
    - **Merge**: `python3 language/operations.py merge <target.md> <source.md> "<Target Node Title>" [--output <out.md>]`
        - Inserts the source tree as children of the specified target node.
    - **Extend**: `python3 language/operations.py extend <target.md> <source.md> [--output <out.md>]`
        - Appends the source tree's top-level items to the target tree's top level.

### 4. `language/migrate.py`
- status: active

- **Purpose**: Heuristically adds default metadata to standard Markdown headers to make them schema-compliant.
- **Usage**: `python3 language/migrate.py <file.md> [file2.md ...]`
- **Effect**: Modifies files in-place by injecting `- status: active` after headers that lack metadata.
### 5. `language/importer.py`
- status: active

- **Purpose**: Converts legacy documents (`.docx`, `.pdf`, `.doc`) into Markdown and auto-applies the Protocol.
- **Usage**: `python3 language/importer.py <file.docx> [file.pdf ...]`
- **Capabilities**:
    - **DOCX**: Preserves headers (Heading 1-3) if `python-docx` is installed. Fallbacks to text extraction.
    - **PDF**: Extracts text if `pypdf` or `pdftotext` is available.
    - **DOC**: Uses MacOS `textutil` for text extraction.
## Migration Guidelines
- status: active

When migrating existing documentation to this schema:
1. **Run the Migration Script**: Use `language/migrate.py` to add baseline metadata.
2. **Review and Refine**: Manually update the `status` fields (e.g., change `active` to `draft` or `deprecated` where appropriate) and add `owner` information.
3. **Structure Check**: Ensure the hierarchy makes sense as a task/node tree.

## Best Practices for AI Generation
- status: active

When generating or modifying files in this repository, AI agents MUST adhere to the following best practices to ensure system stability and parsing accuracy:

1.  **Always Generate IDs**: When creating new nodes (tasks, features, sections), always generate a unique `id` in the metadata (e.g., `id: component.subcomponent.task`). This ensures that references remain stable even if titles change.
2.  **Update Timestamps**: When modifying a node, update the `last_checked` field to the current date (ISO-8601).
3.  **Strict Spacing**: You **MUST** ensure there is exactly one blank line between the metadata block and the content. This is critical for the parser to distinguish between metadata and content lists.
    *   *Correct*:
        ```markdown
        ## Title
        - status: active
        
        Content starts here...
        ```
    *   *Incorrect*:
        ```markdown
        ## Title
        - status: active
        Content starts here...
        ```
4.  **Use Allowed Fields**: Only use metadata keys explicitly listed in the "Allowed Fields" section (`status`, `type`, `owner`, `estimate`, `blocked_by`, `priority`, `id`, `last_checked`) unless you have a specific, documented reason to extend the schema.

