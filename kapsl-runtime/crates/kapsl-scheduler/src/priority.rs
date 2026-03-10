#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    LatencyCritical,
    Throughput,
}
