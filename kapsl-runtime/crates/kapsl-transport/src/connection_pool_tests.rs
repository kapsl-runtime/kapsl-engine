#[cfg(test)]
mod tests {
    use super::super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockConnection {
        id: usize,
    }

    struct MockFactory {
        counter: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl ConnectionFactory for MockFactory {
        type Connection = MockConnection;
        type Error = std::io::Error;

        async fn connect(&self) -> Result<Self::Connection, Self::Error> {
            let id = self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(MockConnection { id })
        }

        async fn is_valid(&self, _conn: &Self::Connection) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn test_pool_lifecycle() {
        let config = PoolConfig {
            max_size: 2,
            ..Default::default()
        };
        let factory = MockFactory {
            counter: AtomicUsize::new(0),
        };
        let pool = ConnectionPool::new(config, factory);

        // 1. Get first connection
        let conn1 = pool.get().await.unwrap();
        assert_eq!(conn1.id, 0);

        let stats = pool.stats().await;
        assert_eq!(stats.active_connections, 1);
        assert_eq!(stats.idle_connections, 0);

        // 2. Get second connection
        let conn2 = pool.get().await.unwrap();
        assert_eq!(conn2.id, 1);

        let stats = pool.stats().await;
        assert_eq!(stats.active_connections, 2);

        // 3. Try third connection (should timeout or fail if we set low timeout)
        // For this test we just verify stats.
        // If we drop conn1, it should go back to pool
        drop(conn1);

        // Give time for async drop to process
        tokio::time::sleep(Duration::from_millis(10)).await;

        let stats = pool.stats().await;
        assert_eq!(stats.active_connections, 1);
        assert_eq!(stats.idle_connections, 1);

        // 4. Reuse connection
        let conn3 = pool.get().await.unwrap();
        assert_eq!(conn3.id, 0); // Should be the reused one
    }
}
