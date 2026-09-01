use std::path::PathBuf;

use clap::Parser;

use super::{Cli, KapslCommand};

#[test]
fn run_accepts_positional_and_backward_compatible_model_paths() {
    let cli = Cli::try_parse_from([
        "kapsl",
        "run",
        "online.aimod",
        "offline.kapsl-bundle",
        "--model",
        "legacy.aimod",
        "--offline",
    ])
    .unwrap();
    let Some(KapslCommand::Run(args)) = cli.command else {
        panic!("expected run command");
    };
    assert_eq!(
        args.input,
        [
            PathBuf::from("online.aimod"),
            PathBuf::from("offline.kapsl-bundle")
        ]
    );
    assert_eq!(args.model, [PathBuf::from("legacy.aimod")]);
    assert!(args.offline);
}

#[test]
fn bundle_accepts_multiple_models_and_cross_target() {
    let cli = Cli::try_parse_from([
        "kapsl",
        "bundle",
        "model-a.aimod",
        "model-b.aimod",
        "--target",
        "linux-x86_64-cuda",
        "--output",
        "production.kapsl-bundle",
    ])
    .unwrap();
    let Some(KapslCommand::Bundle(args)) = cli.command else {
        panic!("expected bundle command");
    };
    assert_eq!(args.model.len(), 2);
    assert_eq!(args.target.as_deref(), Some("linux-x86_64-cuda"));
    assert_eq!(args.output, PathBuf::from("production.kapsl-bundle"));
}
