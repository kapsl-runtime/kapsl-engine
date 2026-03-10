use crate::config::{OptimizationConfig, OptimizationReport};
use crate::error::Result;
use crate::graph::OnnxGraph;
use crate::passes::{
    constant_folding::ConstantFoldingPass, dead_code::DeadCodeEliminationPass,
    fusion::OperatorFusionPass, OptimizationPass,
};
use std::path::Path;
use std::time::Instant;

/// Main graph optimizer
pub struct GraphOptimizer {
    passes: Vec<Box<dyn OptimizationPass>>,
}

impl GraphOptimizer {
    /// Create a new optimizer with the given configuration
    pub fn new(config: OptimizationConfig) -> Self {
        let mut passes: Vec<Box<dyn OptimizationPass>> = Vec::new();

        // Add passes based on configuration
        if config.enable_constant_folding {
            passes.push(Box::new(ConstantFoldingPass));
        }

        if config.enable_fusion {
            passes.push(Box::new(OperatorFusionPass));
        }

        if config.enable_dead_code_elimination {
            passes.push(Box::new(DeadCodeEliminationPass));
        }

        Self { passes }
    }

    /// Optimize a model from input path to output path
    pub fn optimize(&self, input_path: &Path, output_path: &Path) -> Result<OptimizationReport> {
        let start = Instant::now();

        log::info!("Loading ONNX model from {:?}", input_path);
        let mut graph = OnnxGraph::load(input_path)?;

        // Validate input graph
        log::info!("Validating input graph...");
        graph.validate()?;

        let original_nodes = graph.node_count()?;
        log::info!("Original graph has {} nodes", original_nodes);

        // Print node type statistics
        if log::log_enabled!(log::Level::Info) {
            let node_types = graph.get_node_types()?;
            log::info!("Node types:");
            for (op_type, count) in node_types {
                log::info!("  {}: {}", op_type, count);
            }
        }

        // Apply optimization passes
        let mut applied_passes = Vec::new();

        for pass in &self.passes {
            log::info!("Applying pass: {}", pass.name());
            match pass.apply(&mut graph) {
                Ok(modified) => {
                    if modified {
                        applied_passes.push(pass.name().to_string());
                    }
                }
                Err(e) => {
                    log::warn!("Pass {} failed: {}", pass.name(), e);
                }
            }
        }

        // Validate output graph
        log::info!("Validating optimized graph...");
        graph.validate()?;

        let optimized_nodes = graph.node_count()?;
        log::info!("Optimized graph has {} nodes", optimized_nodes);

        // Save optimized model
        log::info!("Saving optimized model to {:?}", output_path);
        graph.save(output_path)?;

        let elapsed = start.elapsed();
        let mut report = OptimizationReport::new(original_nodes, optimized_nodes);
        report.applied_passes = applied_passes;
        report.optimization_time_ms = elapsed.as_millis() as u64;

        log::info!(
            "Optimization complete: {} -> {} nodes ({:.1}% reduction) in {}ms",
            report.original_nodes,
            report.optimized_nodes,
            report.reduction_percent,
            report.optimization_time_ms
        );

        Ok(report)
    }

    /// Optimize a graph in-place
    pub fn optimize_in_place(&self, graph: &mut OnnxGraph) -> Result<OptimizationReport> {
        let start = Instant::now();

        graph.validate()?;
        let original_nodes = graph.node_count()?;

        let mut applied_passes = Vec::new();

        for pass in &self.passes {
            match pass.apply(graph) {
                Ok(modified) => {
                    if modified {
                        applied_passes.push(pass.name().to_string());
                    }
                }
                Err(e) => {
                    log::warn!("Pass {} failed: {}", pass.name(), e);
                }
            }
        }

        graph.validate()?;
        let optimized_nodes = graph.node_count()?;

        let elapsed = start.elapsed();
        let mut report = OptimizationReport::new(original_nodes, optimized_nodes);
        report.applied_passes = applied_passes;
        report.optimization_time_ms = elapsed.as_millis() as u64;

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OptimizationLevel;

    #[test]
    fn test_optimizer_creation() {
        let config = OptimizationConfig::new(OptimizationLevel::O2);
        let optimizer = GraphOptimizer::new(config);

        // Should have passes enabled for O2
        assert!(optimizer.passes.len() > 0);
    }
}
