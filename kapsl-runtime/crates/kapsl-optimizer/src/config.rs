use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    O0,
    /// Basic optimizations (constant folding, dead code elimination)
    O1,
    /// Aggressive optimizations (fusion, reshape elimination)
    O2,
    /// Maximum optimizations (all passes + experimental)
    O3,
}

impl Default for OptimizationLevel {
    fn default() -> Self {
        OptimizationLevel::O1
    }
}

impl std::str::FromStr for OptimizationLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "O0" => Ok(OptimizationLevel::O0),
            "O1" => Ok(OptimizationLevel::O1),
            "O2" => Ok(OptimizationLevel::O2),
            "O3" => Ok(OptimizationLevel::O3),
            _ => Err(format!("Invalid optimization level: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Optimization level
    pub level: OptimizationLevel,

    /// Enable operator fusion
    #[serde(default = "default_true")]
    pub enable_fusion: bool,

    /// Enable constant folding
    #[serde(default = "default_true")]
    pub enable_constant_folding: bool,

    /// Enable dead code elimination
    #[serde(default = "default_true")]
    pub enable_dead_code_elimination: bool,

    /// Enable shape inference
    #[serde(default = "default_true")]
    pub enable_shape_inference: bool,

    /// Target backend (affects optimization strategy)
    pub target_backend: Option<String>,

    /// Custom passes to apply
    #[serde(default)]
    pub custom_passes: Vec<String>,
}

fn default_true() -> bool {
    true
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            level: OptimizationLevel::O1,
            enable_fusion: true,
            enable_constant_folding: true,
            enable_dead_code_elimination: true,
            enable_shape_inference: true,
            target_backend: None,
            custom_passes: vec![],
        }
    }
}

impl OptimizationConfig {
    pub fn new(level: OptimizationLevel) -> Self {
        let mut config = Self::default();
        config.level = level;

        match level {
            OptimizationLevel::O0 => {
                config.enable_fusion = false;
                config.enable_constant_folding = false;
                config.enable_dead_code_elimination = false;
                config.enable_shape_inference = false;
            }
            OptimizationLevel::O1 => {
                config.enable_fusion = false;
                config.enable_constant_folding = true;
                config.enable_dead_code_elimination = true;
                config.enable_shape_inference = true;
            }
            OptimizationLevel::O2 => {
                config.enable_fusion = true;
                config.enable_constant_folding = true;
                config.enable_dead_code_elimination = true;
                config.enable_shape_inference = true;
            }
            OptimizationLevel::O3 => {
                // O3 enables all optimizations including experimental
                config.enable_fusion = true;
                config.enable_constant_folding = true;
                config.enable_dead_code_elimination = true;
                config.enable_shape_inference = true;
            }
        }

        config
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    /// Original node count
    pub original_nodes: usize,

    /// Optimized node count
    pub optimized_nodes: usize,

    /// Reduction percentage
    pub reduction_percent: f32,

    /// Passes that were applied
    pub applied_passes: Vec<String>,

    /// Time taken for optimization (milliseconds)
    pub optimization_time_ms: u64,
}

impl OptimizationReport {
    pub fn new(original_nodes: usize, optimized_nodes: usize) -> Self {
        let reduction_percent = if original_nodes > 0 {
            ((original_nodes - optimized_nodes) as f32 / original_nodes as f32) * 100.0
        } else {
            0.0
        };

        Self {
            original_nodes,
            optimized_nodes,
            reduction_percent,
            applied_passes: vec![],
            optimization_time_ms: 0,
        }
    }
}
