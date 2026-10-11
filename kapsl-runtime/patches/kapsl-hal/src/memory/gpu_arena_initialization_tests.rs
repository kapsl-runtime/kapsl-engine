use super::*;

#[test]
fn failed_initialization_keeps_the_extent_charged_and_unavailable() {
    let kv = PoolOwner::native(1, 0, PoolAllocationClass::KvCache);
    let other = PoolOwner::native(2, 0, PoolAllocationClass::TransientWorkspace);
    let mut policy = PoolPolicy::new(64);
    let mut allocator = AlignedRangeAllocator::new(64);
    let result = allocate_initialized(1, &mut policy, &mut allocator, kv, 16, 1, |_| {
        // Model a submitted zero whose completion fence reports failure.
        Err(ArenaError::InvalidAllocationRequest)
    });
    assert!(matches!(result, Err(ArenaError::InvalidAllocationRequest)));
    let snapshot = build_device_pool_snapshot(&policy, &allocator);
    assert_eq!(snapshot.live_allocation_count, 1);
    assert_eq!(snapshot.allocated_bytes, 16);
    assert_eq!(policy.usage_bytes(kv), 16);
    assert_eq!(allocator.free_bytes(), 48);
    assert!(matches!(
        allocate_initialized(1, &mut policy, &mut allocator, other, 49, 1, |_| Ok(())),
        Err(ArenaError::QuotaExceeded { .. })
    ));
    let live = allocate_initialized(1, &mut policy, &mut allocator, other, 48, 1, |_| {
        panic!("workspace allocations must not run KV initialization")
    })
    .unwrap();
    assert_eq!(live.offset(), 16);
    assert_eq!(policy.usage_bytes(kv), 16);
}

#[test]
fn successful_initialization_publishes_an_extent_that_can_be_explicitly_freed() {
    let kv = PoolOwner::native(1, 0, PoolAllocationClass::KvCache);
    let mut policy = PoolPolicy::new(64);
    let mut allocator = AlignedRangeAllocator::new(64);
    let mut initialized = None;
    let allocation = allocate_initialized(7, &mut policy, &mut allocator, kv, 16, 8, |a| {
        assert_eq!(a.owner(), kv);
        assert_eq!(a.bytes(), 16);
        initialized = Some(a.clone());
        Ok(())
    })
    .unwrap();
    assert_eq!(initialized.unwrap().offset(), allocation.offset());
    assert_eq!(policy.usage_bytes(kv), 16);
    allocator.free(&allocation).unwrap();
    policy.account_free(kv, allocation.bytes());
    assert_eq!(allocator.free_bytes(), 64);
    assert_eq!(policy.usage_bytes(kv), 0);
}

#[test]
fn invalid_requests_and_quota_failure_do_not_submit_initialization() {
    let kv = PoolOwner::native(1, 0, PoolAllocationClass::KvCache);
    let mut policy = PoolPolicy::new(64);
    let mut allocator = AlignedRangeAllocator::new(64);
    policy.set_quota(kv, 0, 8).unwrap();
    for (bytes, alignment) in [(0, 1), (1, 0), (16, 1)] {
        assert!(
            allocate_initialized(1, &mut policy, &mut allocator, kv, bytes, alignment, |_| {
                panic!("rejected allocation reached initialization")
            })
            .is_err()
        );
        assert_eq!(allocator.free_bytes(), 64);
        assert_eq!(policy.usage_bytes(kv), 0);
    }
}

use cudarc::driver::{result, sys, CudaDevice};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

struct Gate {
    release: AtomicBool,
    timed_out: AtomicBool,
}
unsafe extern "C" fn block_default_stream(data: *mut c_void) {
    let gate = unsafe { &*data.cast::<Gate>() };
    let deadline = Instant::now() + Duration::from_secs(5);
    while !gate.release.load(Ordering::Acquire) {
        if Instant::now() >= deadline {
            gate.timed_out.store(true, Ordering::Release);
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}
#[test]
#[ignore = "manual GPU test; the hosted suite checks only host initialization accounting"]
fn kv_zero_completes_on_the_pool_stream_before_publication() {
    assert_eq!(
        std::env::var("CUDA_LAUNCH_BLOCKING").unwrap_or_default(),
        "0"
    );
    let device = CudaDevice::new_with_stream(0).unwrap();
    let pool = GpuDevicePool::new(device.clone(), 1024 * 1024).unwrap();
    device.synchronize().unwrap();
    let workspace = PoolOwner::new(
        PoolBackend::Native,
        0,
        0,
        PoolAllocationClass::TransientWorkspace,
    );
    let kv = PoolOwner::new(PoolBackend::Native, 0, 0, PoolAllocationClass::KvCache);
    pool.set_owner_quota(workspace, 1024 * 1024, 1024 * 1024)
        .unwrap();
    pool.set_owner_admitted(workspace, true).unwrap();
    let dirty = pool.alloc(workspace, 64, 16).unwrap();
    let pointer = pool.allocation_ptr(&dirty) as usize as sys::CUdeviceptr;
    unsafe {
        result::memset_d8_async(pointer, 0xab, 64, *device.cu_stream()).unwrap();
    }
    device.synchronize().unwrap();
    pool.free(dirty).unwrap();
    let mut gate = Box::new(Gate {
        release: AtomicBool::new(false),
        timed_out: AtomicBool::new(false),
    });
    unsafe {
        let driver = sys::lib();
        let mut host = std::ptr::null_mut();
        driver.cuMemHostAlloc(&mut host, 64, 0).result().unwrap();
        driver
            .cuLaunchHostFunc(
                std::ptr::null_mut(),
                Some(block_default_stream),
                (&mut *gate as *mut Gate).cast(),
            )
            .result()
            .unwrap();
        let mut allocation = None;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let started = Instant::now();
            let item = pool.alloc(kv, 64, 16).unwrap();
            let allocation_us = started.elapsed().as_micros();
            let actual_pointer = pool.allocation_ptr(&item) as usize as sys::CUdeviceptr;
            allocation = Some(item);
            assert_eq!(actual_pointer, pointer);
            let default_pending_after_alloc =
                driver.cuStreamQuery(std::ptr::null_mut()) == sys::CUresult::CUDA_ERROR_NOT_READY;
            driver
                .cuMemcpyDtoHAsync_v2(host, pointer, 64, *device.cu_stream())
                .result()
                .unwrap();
            device.synchronize().unwrap();
            let before = std::slice::from_raw_parts(host.cast::<u8>(), 64).to_vec();
            gate.release.store(true, Ordering::Release);
            driver.cuCtxSynchronize().result().unwrap();
            driver
                .cuMemcpyDtoHAsync_v2(host, pointer, 64, *device.cu_stream())
                .result()
                .unwrap();
            device.synchronize().unwrap();
            let after = std::slice::from_raw_parts(host.cast::<u8>(), 64).to_vec();
            println!(
                "{}",
                serde_json::json!({"diagnostic_only":true,"qualification_passed":false,"hal":"workspace candidate","cuda_launch_blocking":false,"allocation_us":allocation_us,"default_pending_after_alloc":default_pending_after_alloc,"gate_timed_out":gate.timed_out.load(Ordering::Acquire),"nonzero_bytes_before_default_completion":before.iter().filter(|b|**b!=0).count(),"dirty_bytes_before_default_completion":before.iter().filter(|b|**b==0xab).count(),"nonzero_bytes_after_default_completion":after.iter().filter(|b|**b!=0).count()})
            );
            assert!(
                !gate.timed_out.load(Ordering::Acquire),
                "control timed out; inconclusive"
            );
            assert!(
                default_pending_after_alloc,
                "allocation waited; publication race not reproduced"
            );
            assert!(
                before.iter().all(|b| *b == 0),
                "allocation published stale bytes before KV zero completion"
            );
            assert!(
                after.iter().all(|b| *b == 0),
                "queued zero did not complete"
            );
        }));
        gate.release.store(true, Ordering::Release);
        driver.cuCtxSynchronize().result().unwrap();
        if let Some(item) = allocation {
            pool.free(item).unwrap();
        }
        driver.cuMemFreeHost(host).result().unwrap();
        if let Err(error) = result {
            std::panic::resume_unwind(error);
        }
    }
}
