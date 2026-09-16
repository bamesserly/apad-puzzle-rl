"""Display utilities for search tree visualization."""


def render_tree(env, tree_state, stats, max_depth=999):
    """Render the search tree visualization.

    Args:
        env: Environment (for piece names)
        tree_state: Dict with 'path', 'subtree_nodes', 'exhausted'
        stats: Statistics dict
        max_depth: Maximum depth to render (default 999 for all)

    Returns:
        String representation of tree
    """
    lines = [
        f"Search Progress ({stats['nodes_explored']} nodes, {stats['backtracks']} backtracks, "
        f"{stats['transposition_hits']} cache hits):"
    ]

    # Group subtree data by depth
    depth_data = {}  # depth -> {piece_id: (nodes, exhausted)}
    for (d, pid), nodes in tree_state["subtree_nodes"].items():
        if d not in depth_data:
            depth_data[d] = {}
        exhausted = tree_state["exhausted"].get((d, pid), False)
        depth_data[d][pid] = (nodes, exhausted)

    # Render tree recursively
    def render_depth(depth, prefix, is_last_child):
        if depth not in depth_data or depth > max_depth:
            return

        pieces_at_depth = list(depth_data[depth].items())
        for i, (piece_id, (nodes, exhausted)) in enumerate(pieces_at_depth):
            is_last = i == len(pieces_at_depth) - 1

            # Determine tree characters
            connector = "└─" if is_last else "├─"

            # Build the line
            piece_name = env.piece_names[piece_id]
            status = (
                "✗"
                if exhausted
                else "(ACTIVE)"
                if depth < len(tree_state["path"]) and tree_state["path"][depth][0] == piece_id
                else "(current)"
            )

            # Check if this piece is in current path
            in_path = depth < len(tree_state["path"]) and tree_state["path"][depth][0] == piece_id

            # Add current action info if at leaf of current path
            extra = ""
            if in_path and depth == len(tree_state["path"]) - 1:
                action_idx, total_actions = (
                    tree_state["path"][depth][1],
                    tree_state["path"][depth][2],
                )
                extra = f" (trying placement {action_idx+1}/{total_actions})"

            line = f"{prefix}{connector} {piece_name}"
            if depth == 0:
                line += f" [root #{piece_id+1}/8]"
            line += f": {nodes} paths {status}{extra}"
            lines.append(line)

            # Recurse to children if this piece is in current path
            if in_path and depth + 1 in depth_data:
                new_prefix = prefix + ("   " if is_last else "│  ")
                render_depth(depth + 1, new_prefix, is_last)

    render_depth(0, "", False)

    # Show untried root pieces
    tried_roots = set(depth_data.get(0, {}).keys())
    untried_count = 8 - len(tried_roots)
    if untried_count > 0:
        lines.append(f"└─ [{untried_count} more root pieces untried]")

    return "\n".join(lines)
