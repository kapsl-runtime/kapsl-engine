pub mod constant_folding;
pub mod dead_code;
pub mod fusion;

use crate::error::Result;
use crate::graph::OnnxGraph;

/// Trait for optimization passes
pub trait OptimizationPass: Send + Sync {
    /// Name of the pass
    fn name(&self) -> &str;

    /// Apply the optimization pass to the graph
    /// Returns true if the graph was modified
    fn apply(&self, graph: &mut OnnxGraph) -> Result<bool>;
}
