use std::collections::HashMap;

/// Node in the Radix Tree
#[derive(Debug, Default)]
pub struct RadixTreeNode {
    /// Map from token ID to child node
    /// Using u64 for token ID to be generic, though typically usually u32
    pub children: HashMap<u64, RadixTreeNode>,

    /// If this node corresponds to the end of a block, this stores the block index
    pub block_index: Option<usize>,
}

impl RadixTreeNode {
    pub fn new() -> Self {
        Self {
            children: HashMap::new(),
            block_index: None,
        }
    }
}

/// Radix Tree for prefix caching of KV cache blocks
#[derive(Debug, Default)]
pub struct RadixTree {
    root: RadixTreeNode,
}

impl RadixTree {
    pub fn new() -> Self {
        Self {
            root: RadixTreeNode::new(),
        }
    }

    /// Insert a block into the tree
    ///
    /// # Arguments
    /// * `tokens` - The sequence of tokens in the block
    /// * `block_index` - The index of the allocated block
    pub fn insert(&mut self, tokens: &[u64], block_index: usize) {
        let mut node = &mut self.root;
        for &token in tokens {
            node = node.children.entry(token).or_default();
        }
        node.block_index = Some(block_index);
    }

    /// Match the longest prefix of tokens to existing blocks
    ///
    /// # Arguments
    /// * `tokens` - The sequence of tokens to match
    ///
    /// # Returns
    /// * `Vec<usize>` - List of block indices that match the prefix
    /// * `usize` - Number of tokens matched
    pub fn match_prefix(&self, tokens: &[u64]) -> (Vec<usize>, usize) {
        let mut node = &self.root;
        let mut blocks = Vec::new();
        let mut matched_len = 0;

        for (i, &token) in tokens.iter().enumerate() {
            if let Some(child) = node.children.get(&token) {
                node = child;

                if let Some(block_idx) = node.block_index {
                    blocks.push(block_idx);
                    matched_len = i + 1;
                }
            } else {
                break;
            }
        }

        (blocks, matched_len)
    }

    /// Remove a block reference from the tree
    /// This is complex because we need to find the node.
    /// Typically we just lazy delete or let LRU handle it?
    /// For strict ref counting, we might want to support removal.
    /// But removal requires traversing to the leaf.
    ///
    /// For now, we assume blocks are managed by the allocator and the tree
    /// is just an index. If a block is freed and reused, we should probably
    /// invalidate the tree entry.
    ///
    /// Optimization: Store a mapping from block_index -> node path?
    /// Or just clear the tree when clearing cache?
    ///
    /// Let's add a simple remove that takes tokens (since we know the tokens for a block usually)
    pub fn remove(&mut self, tokens: &[u64]) {
        // Recursive remove is cleaner for cleanup
        Self::remove_recursive(&mut self.root, tokens, 0);
    }

    fn remove_recursive(node: &mut RadixTreeNode, tokens: &[u64], idx: usize) -> bool {
        if idx == tokens.len() {
            node.block_index = None;
            return node.children.is_empty();
        }

        let token = tokens[idx];
        if let Some(child) = node.children.get_mut(&token) {
            if Self::remove_recursive(child, tokens, idx + 1) {
                node.children.remove(&token);
                return node.children.is_empty() && node.block_index.is_none();
            }
        }
        false
    }
}
