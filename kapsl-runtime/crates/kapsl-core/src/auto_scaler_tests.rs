#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_scale_up_decision() {
        let mut scaler = AutoScaler::new();
        let policy = ScalingPolicy {
            min_replicas: 1,
            max_replicas: 4,
            target_queue_depth: 5,
            scale_down_threshold: 2,
            cooldown_seconds: 300,
        };
        scaler.register_policy(0, policy);

        // First sample is below hysteresis window.
        let result = scaler.should_scale_up(0, 1, 1, 20, Duration::from_secs(10), true);
        assert!(result.is_none());

        // Sustained high queue depth triggers bounded step-up (+1).
        let result = scaler.should_scale_up(0, 1, 1, 20, Duration::from_secs(10), true);
        assert_eq!(result, Some(2));
    }

    #[test]
    fn test_max_replicas_limit() {
        let mut scaler = AutoScaler::new();
        let policy = ScalingPolicy {
            min_replicas: 1,
            max_replicas: 2,
            target_queue_depth: 5,
            scale_down_threshold: 2,
            cooldown_seconds: 300,
        };
        scaler.register_policy(0, policy);

        // Already at max replicas
        let result = scaler.should_scale_up(0, 2, 2, 20, Duration::from_secs(30), true);
        assert!(result.is_none());
    }

    #[test]
    fn test_scale_up_with_fractional_average() {
        let mut scaler = AutoScaler::new();
        let policy = ScalingPolicy {
            min_replicas: 1,
            max_replicas: 4,
            target_queue_depth: 5,
            scale_down_threshold: 2,
            cooldown_seconds: 300,
        };
        scaler.register_policy(0, policy);

        // 11/2 = 5.5, so we should still scale up, capped to +1 step.
        let result = scaler.should_scale_up(0, 2, 2, 11, Duration::from_secs(30), true);
        assert_eq!(result, Some(3));
    }

    #[test]
    fn test_scale_down_requires_cooldown() {
        let mut scaler = AutoScaler::new();
        let policy = ScalingPolicy {
            min_replicas: 1,
            max_replicas: 4,
            target_queue_depth: 5,
            scale_down_threshold: 2,
            cooldown_seconds: 1, // 1 second for testing
        };
        scaler.register_policy(0, policy);

        // Low queue depth, but not enough time has passed
        let result = scaler.should_scale_down(0, 3, 3, 1, Duration::from_millis(500), true);
        assert!(result.is_none());

        // After cooldown period
        let result = scaler.should_scale_down(0, 3, 3, 1, Duration::from_secs(1), true);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), 2); // Scale down by 1
    }

    #[test]
    fn test_min_replicas_limit() {
        let mut scaler = AutoScaler::new();
        let policy = ScalingPolicy {
            min_replicas: 2,
            max_replicas: 4,
            target_queue_depth: 5,
            scale_down_threshold: 2,
            cooldown_seconds: 1,
        };
        scaler.register_policy(0, policy);

        // Already at min replicas
        let result = scaler.should_scale_down(0, 2, 2, 0, Duration::from_secs(2), true);
        assert!(result.is_none());
    }

    #[test]
    fn test_get_next_replica_id() {
        let scaler = AutoScaler::new();

        // No existing replicas
        assert_eq!(scaler.get_next_replica_id(0, &[]), 1);

        // With existing replicas
        assert_eq!(scaler.get_next_replica_id(0, &[0, 1, 2]), 3);
    }

    #[test]
    fn test_cooldown_prevention() {
        let mut scaler = AutoScaler::new();
        let policy = ScalingPolicy {
            min_replicas: 1,
            max_replicas: 4,
            target_queue_depth: 5,
            scale_down_threshold: 2,
            cooldown_seconds: 10,
        };
        scaler.register_policy(0, policy);

        // First scale-down should work after cooldown
        let result = scaler.should_scale_down(0, 3, 3, 1, Duration::from_secs(11), true);
        assert!(result.is_some());

        // Immediate second scale-down should be prevented
        let result = scaler.should_scale_down(0, 2, 2, 1, Duration::from_secs(1), true);
        assert!(result.is_none());
    }

    #[test]
    fn test_scaling_policy_validation() {
        let valid = ScalingPolicy {
            min_replicas: 1,
            max_replicas: 4,
            target_queue_depth: 5,
            scale_down_threshold: 2,
            cooldown_seconds: 300,
        };
        assert!(valid.validate().is_ok());

        let invalid_min = ScalingPolicy {
            min_replicas: 0,
            ..valid.clone()
        };
        assert!(invalid_min.validate().is_err());

        let invalid_bounds = ScalingPolicy {
            min_replicas: 3,
            max_replicas: 2,
            ..valid.clone()
        };
        assert!(invalid_bounds.validate().is_err());

        let invalid_target = ScalingPolicy {
            target_queue_depth: 0,
            ..valid.clone()
        };
        assert!(invalid_target.validate().is_err());

        let invalid_cooldown = ScalingPolicy {
            cooldown_seconds: 0,
            ..valid
        };
        assert!(invalid_cooldown.validate().is_err());
    }

    #[test]
    fn test_zero_target_queue_depth_is_ignored_for_scale_up() {
        let mut scaler = AutoScaler::new();
        let policy = ScalingPolicy {
            min_replicas: 1,
            max_replicas: 4,
            target_queue_depth: 0,
            scale_down_threshold: 2,
            cooldown_seconds: 300,
        };
        scaler.register_policy(0, policy);

        // Defensive behavior: invalid policy does not panic or force a bogus scale-up.
        let result = scaler.should_scale_up(0, 1, 1, 100, Duration::from_secs(30), true);
        assert!(result.is_none());
    }

    #[test]
    fn test_scale_up_blocked_when_replicas_unhealthy() {
        let mut scaler = AutoScaler::new();
        let policy = ScalingPolicy {
            min_replicas: 1,
            max_replicas: 4,
            target_queue_depth: 5,
            scale_down_threshold: 2,
            cooldown_seconds: 300,
        };
        scaler.register_policy(0, policy);

        // One active replica is unhealthy, so we should not scale up from noisy signals.
        let result = scaler.should_scale_up(0, 2, 1, 50, Duration::from_secs(30), true);
        assert!(result.is_none());
    }

    #[test]
    fn test_scale_down_blocked_when_metrics_unavailable() {
        let mut scaler = AutoScaler::new();
        let policy = ScalingPolicy {
            min_replicas: 1,
            max_replicas: 4,
            target_queue_depth: 5,
            scale_down_threshold: 2,
            cooldown_seconds: 1,
        };
        scaler.register_policy(0, policy);

        let result = scaler.should_scale_down(0, 3, 3, 0, Duration::from_secs(5), false);
        assert!(result.is_none());
    }
}
