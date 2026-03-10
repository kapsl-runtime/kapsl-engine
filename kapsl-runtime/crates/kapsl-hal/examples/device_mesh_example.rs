// Example: Using Device Mesh for Distributed Inference
//
// This example demonstrates how to use the Device Mesh API for
// distributing ML inference across multiple devices.

use kapsl_hal::device::{Device, DeviceBackend, DeviceInfo};
use kapsl_hal::device_mesh::{DeviceMesh, GroupBackend, MeshTopology};

fn main() {
    println!("🌐 Device Mesh Example\n");

    // 1. Probe available devices
    let device_info = DeviceInfo::probe();
    println!("📊 System Info:");
    println!("   CPU Cores: {}", device_info.cpu_cores);
    println!(
        "   Total Memory: {} MB",
        device_info.total_memory / 1024 / 1024
    );
    println!(
        "   CUDA: {}",
        if device_info.has_cuda { "✅" } else { "❌" }
    );
    println!(
        "   Metal: {}",
        if device_info.has_metal { "✅" } else { "❌" }
    );
    println!(
        "   ROCm: {}",
        if device_info.has_rocm { "✅" } else { "❌" }
    );
    println!();

    // 2. Create a simple Data Parallel mesh
    println!("Example 1: Data Parallel Mesh");
    println!("{}", "-".repeat(50));

    let devices = device_info.devices.clone();
    let mesh = DeviceMesh::new(devices);

    println!("   World Size: {}", mesh.world_size);
    println!("   Topology: {:?}", mesh.topology);
    println!("   Groups: {:?}", mesh.group_names());

    let stats = mesh.stats();
    println!("   Total Memory: {} MB", stats.total_memory_mb);
    println!("   Total Compute Units: {}", stats.total_compute_units);
    println!("   Backend Distribution: {:?}", stats.backend_distribution);
    println!();

    // 3. Create a Tensor Parallel mesh with multiple GPUs
    if device_info.has_cuda && device_info.cuda_devices().len() >= 4 {
        println!("Example 2: Tensor Parallel Mesh (4 GPUs)");
        println!("{}", "-".repeat(50));

        // Create mock CUDA devices for demonstration
        let cuda_devices: Vec<Device> = (0..4)
            .map(|i| Device {
                id: i,
                name: format!("NVIDIA GPU {}", i),
                backend: DeviceBackend::Cuda,
                memory_mb: 16000,
                compute_units: 80,
                pci_bus_id: None,
                driver_version: None,
                compute_capability: None,
                utilization_gpu_pct: None,
                temperature_c: None,
                supports_fp16: true,
                supports_int8: true,
                cuda_version: Some("12.0".to_string()),
            })
            .collect();

        let tp_topology = MeshTopology::TensorParallel {
            degree: 4,
            mesh_shape: (1, 4),
        };

        match DeviceMesh::with_topology(cuda_devices, tp_topology) {
            Ok(tp_mesh) => {
                println!("   ✅ Tensor Parallel Mesh created");
                println!("   World Size: {}", tp_mesh.world_size);
                println!("   Groups: {:?}", tp_mesh.group_names());

                // Access TP group
                if let Some(tp_group) = tp_mesh.get_group("tp_0") {
                    println!("   TP Group 0 ranks: {:?}", tp_group.ranks);
                    println!("   Backend: {:?}", tp_group.backend);
                }
            }
            Err(e) => println!("   ❌ Failed to create TP mesh: {}", e),
        }
        println!();
    }

    // 4. Create a Mixed Parallelism mesh
    println!("Example 3: Mixed Parallelism (TP=2, PP=2, DP=2)");
    println!("{}", "-".repeat(50));

    let mixed_devices: Vec<Device> = (0..8)
        .map(|i| Device {
            id: i,
            name: format!("GPU {}", i),
            backend: DeviceBackend::Cuda,
            memory_mb: 24000,
            compute_units: 108,
            pci_bus_id: None,
            driver_version: None,
            compute_capability: None,
            utilization_gpu_pct: None,
            temperature_c: None,
            supports_fp16: true,
            supports_int8: true,
            cuda_version: Some("12.0".to_string()),
        })
        .collect();

    let mixed_topology = MeshTopology::Mixed {
        tp: 2, // Tensor Parallel degree
        pp: 2, // Pipeline Parallel stages
        dp: 2, // Data Parallel replicas
    };

    match DeviceMesh::with_topology(mixed_devices, mixed_topology) {
        Ok(mesh) => {
            println!("   ✅ Mixed Parallel Mesh created");
            println!("   World Size: {}", mesh.world_size);
            println!("   Number of groups: {}", mesh.group_names().len());
            println!("   Groups:");
            for group_name in mesh.group_names() {
                if let Some(group) = mesh.get_group(&group_name) {
                    println!("      - {}: ranks {:?}", group_name, group.ranks);
                }
            }
        }
        Err(e) => println!("   ❌ Failed to create mixed mesh: {}", e),
    }
    println!();

    // 5. Demonstrate custom group creation
    println!("Example 4: Custom Process Groups");
    println!("{}", "-".repeat(50));

    let devices = (0..4)
        .map(|i| Device {
            id: i,
            name: format!("Device {}", i),
            backend: DeviceBackend::Cpu,
            memory_mb: 8000,
            compute_units: 4,
            pci_bus_id: None,
            driver_version: None,
            compute_capability: None,
            utilization_gpu_pct: None,
            temperature_c: None,
            supports_fp16: true,
            supports_int8: true,
            cuda_version: None,
        })
        .collect();

    let mut mesh = DeviceMesh::new(devices);

    // Add custom groups
    mesh.add_group("replicas_a".to_string(), vec![0, 1], GroupBackend::Gloo)
        .unwrap();

    mesh.add_group("replicas_b".to_string(), vec![2, 3], GroupBackend::Gloo)
        .unwrap();

    println!("   ✅ Custom groups created");
    println!("   Groups: {:?}", mesh.group_names());

    // Check group membership
    mesh.set_rank(1).unwrap();
    println!("   Rank 1 in 'replicas_a': {}", mesh.in_group("replicas_a"));
    println!("   Rank 1 in 'replicas_b': {}", mesh.in_group("replicas_b"));
    println!(
        "   Local rank in 'replicas_a': {:?}",
        mesh.group_rank("replicas_a")
    );

    println!("\n✅ Device Mesh examples completed!");
}
