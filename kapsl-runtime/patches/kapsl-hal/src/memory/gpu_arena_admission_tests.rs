use super::*;

#[test]
fn admission_query_tracks_workloads_without_granting_borrowers_admission() {
    let weights = PoolOwner::onnx(1, 0, PoolAllocationClass::PersistentWeights);
    let workspace = PoolOwner::onnx(1, 0, PoolAllocationClass::TransientWorkspace);
    let replica = PoolOwner::onnx(1, 1, PoolAllocationClass::TransientWorkspace);
    let model = PoolOwner::onnx(2, 0, PoolAllocationClass::TransientWorkspace);
    let backend = PoolOwner::gguf(1, 0, PoolAllocationClass::TransientWorkspace);
    let unattributed =
        PoolOwner::unattributed(PoolBackend::Onnx, PoolAllocationClass::TransientWorkspace);
    let mut policy = PoolPolicy::new(1024);
    let mut allocator = AlignedRangeAllocator::new(1024);
    policy.set_quota(weights, 128, 512).unwrap();
    assert!(!policy.is_admitted(weights), "a quota is not admission");
    policy.set_admitted(weights, true);
    for owner in [weights, workspace] {
        assert!(
            policy.is_admitted(owner),
            "classes share workload admission"
        );
    }
    for owner in [replica, model, backend, unattributed] {
        assert!(
            !policy.is_admitted(owner),
            "admission cannot cross workloads"
        );
    }

    // The legacy allocator deliberately permits elastic borrowing. Its usage
    // entry must not be mistaken for engine authorization by the new query.
    let borrowed = allocate_initialized(1, &mut policy, &mut allocator, replica, 64, 16, |_| {
        panic!("workspace borrowing must not initialize KV memory")
    })
    .unwrap();
    assert_eq!(policy.usage_bytes(replica), 64);
    assert!(!policy.is_admitted(replica));
    allocator.free(&borrowed).unwrap();
    policy.account_free(replica, borrowed.bytes());

    policy.set_admitted(weights, false);
    assert!(!policy.is_admitted(weights));
    assert!(!policy.is_admitted(workspace));
    assert_eq!(policy.quota(weights).guaranteed_bytes, 128);
    assert!(build_device_pool_snapshot(&policy, &allocator)
        .owners
        .is_empty());
}

/// CPU-only diagnostic of the old engine admission lookup and the narrow
/// query. Includes their actual policy/allocator lock patterns; creates no
/// CUDA device. This is not an end-to-end latency or GPU qualification test.
#[test]
#[ignore = "manual host timing diagnostic; correctness CI must not gate on timing"]
fn snapshot_admission_timing_diagnostic() {
    use std::hint::black_box;
    use std::time::Instant;

    let mut cases = Vec::new();
    for workloads in [1_u32, 2, 8, 32] {
        let mut policy = PoolPolicy::new(64 * 1024 * 1024);
        let mut allocator = AlignedRangeAllocator::new(64 * 1024 * 1024);
        for model in 0..workloads {
            let weights = PoolOwner::onnx(model, 0, PoolAllocationClass::PersistentWeights);
            policy.set_quota(weights, 1024, 64 * 1024 * 1024).unwrap();
            policy.set_admitted(weights, true);
            for class in [
                PoolAllocationClass::PersistentWeights,
                PoolAllocationClass::TransientWorkspace,
            ] {
                let owner = PoolOwner::onnx(model, 0, class);
                let allocation = allocator.alloc(1, owner, 256, 16).unwrap();
                policy.account_alloc(owner, allocation.bytes());
            }
        }
        let policy = Mutex::new(policy);
        let allocator = Mutex::new(allocator);
        let owner = PoolOwner::onnx(0, 0, PoolAllocationClass::TransientWorkspace);
        let iterations = if workloads <= 2 { 20_000 } else { 2_000 };
        let mut samples = Vec::new();
        for block in 0..3 {
            for snapshot in [true, false, false, true] {
                let started = Instant::now();
                for _ in 0..iterations {
                    let owner = black_box(owner);
                    let admitted = if snapshot {
                        let policy = policy.lock().unwrap();
                        let allocator = allocator.lock().unwrap();
                        let state = black_box(build_device_pool_snapshot(&policy, &allocator));
                        state.owners.iter().any(|entry| {
                            entry.owner.workload() == owner.workload() && entry.admitted
                        })
                    } else {
                        policy.lock().unwrap().is_admitted(owner)
                    };
                    assert!(black_box(admitted));
                }
                samples.push(serde_json::json!({
                    "block": block,
                    "variant": if snapshot { "snapshot" } else { "admission_query" },
                    "ns_per_lookup": started.elapsed().as_nanos() as f64 / iterations as f64,
                }));
            }
        }
        cases.push(serde_json::json!({"workloads":workloads,"live_owners":workloads*2,"iterations_per_window":iterations,"samples":samples}));
    }
    println!(
        "{}",
        serde_json::json!({"diagnostic_only":true,"qualification_passed":false,"architecture":std::env::consts::ARCH,"cases":cases})
    );
}
