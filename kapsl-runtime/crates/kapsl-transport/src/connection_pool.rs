use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};

/// Configuration for the connection pool
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Minimum number of idle connections to maintain
    pub min_idle: usize,
    /// Maximum number of connections in the pool
    pub max_size: usize,
    /// Maximum time a connection can be idle before being closed
    pub idle_timeout: Duration,
    /// Maximum time to wait for a connection
    pub connection_timeout: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_idle: 1,
            max_size: 10,
            idle_timeout: Duration::from_secs(60),
            connection_timeout: Duration::from_secs(5),
        }
    }
}

/// Statistics for the connection pool
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub active_connections: usize,
    pub idle_connections: usize,
    pub total_connections: usize,
}

/// A generic connection pool for managing network connections
pub struct ConnectionPool<C, F>
where
    C: Send + Sync + 'static,
    F: ConnectionFactory<Connection = C> + Send + Sync + 'static,
{
    config: PoolConfig,
    // Semaphore to limit maximum concurrent connections
    sem: Arc<Semaphore>,
    // Queue of idle connections
    idle_connections: Arc<Mutex<VecDeque<IdleConnection<C>>>>,
    // Factory for creating new connections
    factory: Arc<F>,
}

struct IdleConnection<C> {
    connection: C,
    last_used: Instant,
}

/// Trait for creating and validating connections
#[async_trait::async_trait]
pub trait ConnectionFactory {
    type Connection: Send + Sync + 'static;
    type Error: std::error::Error + Send + Sync + 'static;

    async fn connect(&self) -> Result<Self::Connection, Self::Error>;
    async fn is_valid(&self, conn: &Self::Connection) -> bool;
}

/// A handle to a pooled connection
pub struct PooledConnection<C, F>
where
    C: Send + Sync + 'static,
    F: ConnectionFactory<Connection = C> + Send + Sync + 'static,
{
    pool: Arc<ConnectionPool<C, F>>,
    connection: Option<C>,
}

impl<C, F> std::ops::Deref for PooledConnection<C, F>
where
    C: Send + Sync + 'static,
    F: ConnectionFactory<Connection = C> + Send + Sync + 'static,
{
    type Target = C;

    fn deref(&self) -> &Self::Target {
        self.connection.as_ref().unwrap()
    }
}

impl<C, F> std::ops::DerefMut for PooledConnection<C, F>
where
    C: Send + Sync + 'static,
    F: ConnectionFactory<Connection = C> + Send + Sync + 'static,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.connection.as_mut().unwrap()
    }
}

impl<C, F> Drop for PooledConnection<C, F>
where
    C: Send + Sync + 'static,
    F: ConnectionFactory<Connection = C> + Send + Sync + 'static,
{
    fn drop(&mut self) {
        if let Some(conn) = self.connection.take() {
            let pool = self.pool.clone();
            tokio::spawn(async move {
                pool.return_connection(conn).await;
            });
        }
    }
}

impl<C, F> ConnectionPool<C, F>
where
    C: Send + Sync + 'static,
    F: ConnectionFactory<Connection = C> + Send + Sync + 'static,
{
    pub fn new(config: PoolConfig, factory: F) -> Self {
        Self {
            sem: Arc::new(Semaphore::new(config.max_size)),
            idle_connections: Arc::new(Mutex::new(VecDeque::new())),
            factory: Arc::new(factory),
            config,
        }
    }

    /// Acquire a connection from the pool
    pub async fn get(
        &self,
    ) -> Result<PooledConnection<C, F>, Box<dyn std::error::Error + Send + Sync>> {
        // Try to get an idle connection first
        {
            let mut idle = self.idle_connections.lock().await;
            while let Some(idle_conn) = idle.pop_front() {
                if idle_conn.last_used.elapsed() > self.config.idle_timeout {
                    // Connection expired, drop it and continue
                    continue;
                }

                if self.factory.is_valid(&idle_conn.connection).await {
                    return Ok(PooledConnection {
                        pool: Arc::new(self.clone()),
                        connection: Some(idle_conn.connection),
                    });
                }
            }
        }

        // No valid idle connection, create a new one
        // Wait for a permit
        let permit_result =
            tokio::time::timeout(self.config.connection_timeout, self.sem.acquire()).await;

        let permit = match permit_result {
            Ok(Ok(p)) => p,
            Ok(Err(_)) => return Err("Pool closed".into()),
            Err(_) => return Err("Connection timeout waiting for pool slot".into()),
        };

        // We have a permit, create the connection
        let conn = self
            .factory
            .connect()
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

        // We need to "forget" the permit because the PooledConnection destructor
        // or return_connection logic will handle the lifecycle.
        permit.forget();

        Ok(PooledConnection {
            pool: Arc::new(self.clone()),
            connection: Some(conn),
        })
    }

    async fn return_connection(&self, conn: C) {
        // Check if connection is still valid
        if self.factory.is_valid(&conn).await {
            let mut idle = self.idle_connections.lock().await;
            if idle.len() < self.config.max_size {
                idle.push_back(IdleConnection {
                    connection: conn,
                    last_used: Instant::now(),
                });
                return;
            }
        }

        // If we reach here, we are discarding the connection
        // Release the semaphore permit
        self.sem.add_permits(1);
    }

    pub async fn stats(&self) -> PoolStats {
        let idle = self.idle_connections.lock().await.len();
        let available_permits = self.sem.available_permits();
        let total_limit = self.config.max_size;
        let active = total_limit - available_permits - idle;

        PoolStats {
            active_connections: active,
            idle_connections: idle,
            total_connections: active + idle,
        }
    }

    // Helper to clone self for the PooledConnection
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            sem: self.sem.clone(),
            idle_connections: self.idle_connections.clone(),
            factory: self.factory.clone(),
        }
    }
}

#[cfg(test)]
#[path = "connection_pool_tests.rs"]
mod connection_pool_tests;
