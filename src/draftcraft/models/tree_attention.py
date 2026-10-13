"""Tree attention mask construction for speculative candidate verification."""

from typing import List, Optional, Tuple
import torch


class TreeAttentionBuilder:
    """Builds custom 2D attention masks for tree-based speculative decoding."""

    @staticmethod
    def build_tree_mask(
        tree_paths: List[List[int]],
        prefix_len: int = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Construct an attention mask for a speculative candidate tree.

        Each candidate token in the tree can attend only to:
        1. All prefix context tokens (0 ... prefix_len - 1).
        2. Its own ancestor tokens within the draft tree path.

        Args:
            tree_paths: List of token index paths from tree root to leaves.
                        e.g. [[0, 1, 2], [0, 1, 3], [0, 4]] where numbers are relative candidate indices.
            prefix_len: Length of existing context prompt tokens.
            device: Target torch device.

        Returns:
            Boolean attention mask tensor of shape [Total_Tokens, Total_Tokens],
            where True indicates allowed attention and False indicates masked attention.
        """
        # Determine total unique candidate nodes in tree
        all_nodes = set()
        for path in tree_paths:
            all_nodes.update(path)

        num_candidates = len(all_nodes) if all_nodes else 0
        total_len = prefix_len + num_candidates

        # Initialize mask with zeros (masked)
        mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)

        # 1. Prefix tokens have standard causal attention among themselves
        if prefix_len > 0:
            prefix_causal = torch.tril(torch.ones((prefix_len, prefix_len), dtype=torch.bool, device=device))
            mask[:prefix_len, :prefix_len] = prefix_causal

        # 2. Every candidate node attends to ALL prefix tokens
        if prefix_len > 0 and num_candidates > 0:
            mask[prefix_len:, :prefix_len] = True

        # 3. Each candidate node attends to its ancestors along valid tree paths
        for path in tree_paths:
            for i, node in enumerate(path):
                cand_idx = prefix_len + node
                # Can attend to all preceding nodes in this path
                for ancestor in path[: i + 1]:
                    anc_idx = prefix_len + ancestor
                    mask[cand_idx, anc_idx] = True

        return mask

    @staticmethod
    def generate_balanced_tree_paths(
        depth: int,
        branching_factor: int,
    ) -> Tuple[List[List[int]], int]:
        """Generate balanced speculative tree paths.

        Returns:
            Tuple of (paths_list, total_candidate_nodes).
        """
        paths: List[List[int]] = []
        current_node_id = 0

        def build_branch(curr_path: List[int], current_depth: int) -> None:
            nonlocal current_node_id
            if current_depth == depth:
                paths.append(curr_path)
                return

            for _ in range(branching_factor):
                next_node = current_node_id
                current_node_id += 1
                build_branch(curr_path + [next_node], current_depth + 1)

        build_branch([], 0)
        return paths, current_node_id
