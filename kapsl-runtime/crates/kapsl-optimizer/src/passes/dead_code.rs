use crate::error::Result;
use crate::graph::OnnxGraph;
use crate::passes::OptimizationPass;

/// Dead code elimination pass
/// Removes nodes that don't contribute to the final output
pub struct DeadCodeEliminationPass;

impl OptimizationPass for DeadCodeEliminationPass {
    fn name(&self) -> &str {
        "dead_code_elimination"
    }

    fn apply(&self, graph: &mut OnnxGraph) -> Result<bool> {
        let removed = graph.remove_unused_nodes()?;

        if removed > 0 {
            log::info!("Dead code elimination: removed {} unused nodes", removed);
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
