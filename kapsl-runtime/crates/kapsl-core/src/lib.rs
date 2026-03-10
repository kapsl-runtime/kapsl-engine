pub mod auto_scaler;
pub mod loader;
pub mod model_registry;
pub mod requirements;

pub use auto_scaler::{AutoScaler, ScalingPolicy};
pub use loader::PackageLoader;
pub use model_registry::{ModelInfo, ModelRegistry, ModelStatus};
pub use requirements::HardwareRequirements;
