//! Resolution of CLI performance profiles.

use super::*;
use clap::{parser::ValueSource, ArgMatches};

fn arg_user_supplied(matches: &ArgMatches, arg: &str) -> bool {
    !matches!(
        matches.value_source(arg),
        None | Some(ValueSource::DefaultValue)
    )
}

pub(crate) fn apply_performance_profile(
    args: &mut Args,
    matches: &ArgMatches,
) -> AppliedPerformanceTuning {
    let mut tuning = AppliedPerformanceTuning::default();
    if args.worker || matches!(args.performance_profile, PerformanceProfile::Standard) {
        return tuning;
    }

    let batch_explicit = arg_user_supplied(matches, "batch_size");
    let transport_explicit = arg_user_supplied(matches, "transport");
    let scheduler_queue_size_explicit = arg_user_supplied(matches, "scheduler_queue_size");
    let scheduler_max_micro_batch_explicit =
        arg_user_supplied(matches, "scheduler_max_micro_batch");
    let scheduler_queue_delay_ms_explicit = arg_user_supplied(matches, "scheduler_queue_delay_ms");

    match args.performance_profile {
        PerformanceProfile::Standard => {}
        PerformanceProfile::Auto => {
            let policy = auto_tune_policy(&args.model);
            if !batch_explicit {
                args.batch_size = policy.batch_size;
                tuning.batch_size = Some(args.batch_size);
            }
            if !scheduler_queue_size_explicit {
                args.scheduler_queue_size = policy.scheduler_queue_size;
                tuning.scheduler_queue_size = Some(args.scheduler_queue_size);
            }
            if !scheduler_max_micro_batch_explicit {
                args.scheduler_max_micro_batch = policy.scheduler_max_micro_batch;
                tuning.scheduler_max_micro_batch = Some(args.scheduler_max_micro_batch);
            }
            if !scheduler_queue_delay_ms_explicit {
                args.scheduler_queue_delay_ms = policy.scheduler_queue_delay_ms;
                tuning.scheduler_queue_delay_ms = Some(args.scheduler_queue_delay_ms);
            }
            tuning.auto_tune_rationale = Some(format!(
                "batch={}, micro_batch={}, delay={}ms, queue_size={} | {}",
                args.batch_size,
                args.scheduler_max_micro_batch,
                args.scheduler_queue_delay_ms,
                args.scheduler_queue_size,
                policy.rationale,
            ));
        }
        PerformanceProfile::Balanced => {
            if !batch_explicit {
                args.batch_size = 8;
                tuning.batch_size = Some(args.batch_size);
            }
            if !transport_explicit {
                args.transport = "hybrid".to_string();
                tuning.transport = Some(args.transport.clone());
            }
            if !scheduler_queue_size_explicit {
                args.scheduler_queue_size = 512;
                tuning.scheduler_queue_size = Some(args.scheduler_queue_size);
            }
            if !scheduler_max_micro_batch_explicit {
                args.scheduler_max_micro_batch = args.batch_size.max(1);
                tuning.scheduler_max_micro_batch = Some(args.scheduler_max_micro_batch);
            }
            if !scheduler_queue_delay_ms_explicit {
                args.scheduler_queue_delay_ms = 3;
                tuning.scheduler_queue_delay_ms = Some(args.scheduler_queue_delay_ms);
            }
        }
        PerformanceProfile::Throughput => {
            if !batch_explicit {
                args.batch_size = 16;
                tuning.batch_size = Some(args.batch_size);
            }
            if !transport_explicit {
                args.transport = "hybrid".to_string();
                tuning.transport = Some(args.transport.clone());
            }
            if !scheduler_queue_size_explicit {
                args.scheduler_queue_size = 2048;
                tuning.scheduler_queue_size = Some(args.scheduler_queue_size);
            }
            if !scheduler_max_micro_batch_explicit {
                args.scheduler_max_micro_batch = args.batch_size.max(1);
                tuning.scheduler_max_micro_batch = Some(args.scheduler_max_micro_batch);
            }
            if !scheduler_queue_delay_ms_explicit {
                args.scheduler_queue_delay_ms = 6;
                tuning.scheduler_queue_delay_ms = Some(args.scheduler_queue_delay_ms);
            }
            if std::env::var_os("RUST_LOG").is_none() {
                std::env::set_var("RUST_LOG", "warn");
                tuning.rust_log = Some("warn".to_string());
            }
        }
        PerformanceProfile::Latency => {
            if !batch_explicit {
                args.batch_size = 1;
                tuning.batch_size = Some(args.batch_size);
            }
            if !transport_explicit {
                args.transport = "socket".to_string();
                tuning.transport = Some(args.transport.clone());
            }
            if !scheduler_queue_size_explicit {
                args.scheduler_queue_size = 128;
                tuning.scheduler_queue_size = Some(args.scheduler_queue_size);
            }
            if !scheduler_max_micro_batch_explicit {
                args.scheduler_max_micro_batch = 1;
                tuning.scheduler_max_micro_batch = Some(args.scheduler_max_micro_batch);
            }
            if !scheduler_queue_delay_ms_explicit {
                args.scheduler_queue_delay_ms = 0;
                tuning.scheduler_queue_delay_ms = Some(args.scheduler_queue_delay_ms);
            }
        }
    }

    tuning
}
