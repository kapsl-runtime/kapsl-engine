//! Backend-owned registrations attached to one physical device pool.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

pub(crate) type PoolClientCleanup = Box<dyn Fn() -> Result<(), String> + Send + Sync>;

struct Registration {
    users: usize,
    cleanup: PoolClientCleanup,
}

#[derive(Default)]
pub(crate) struct PoolClients {
    registrations: Mutex<BTreeMap<&'static str, Registration>>,
}

impl PoolClients {
    pub(crate) fn acquire(
        self: &Arc<Self>,
        client: &'static str,
        register: impl FnOnce() -> Result<PoolClientCleanup, String>,
    ) -> Result<PoolClientLease, String> {
        let mut registrations = self.registrations.lock().unwrap();
        if let Some(registration) = registrations.get_mut(client) {
            if registration.users > 0 {
                registration.users = registration.users.checked_add(1).ok_or_else(|| {
                    format!("device pool client {client} reference count overflow")
                })?;
                return Ok(PoolClientLease::new(self.clone(), client));
            }
            // A failed terminal cleanup must complete before another model
            // can reuse the registration. Keep its storage on failure.
            (registration.cleanup)()?;
            registrations.remove(client);
        }
        let cleanup = register()?;
        registrations.insert(client, Registration { users: 1, cleanup });
        Ok(PoolClientLease::new(self.clone(), client))
    }

    fn release(&self, client: &'static str) {
        let mut registrations = self.registrations.lock().unwrap();
        let registration = registrations
            .get_mut(client)
            .expect("a pool client lease retains its registration");
        registration.users -= 1;
        if registration.users == 0 {
            match (registration.cleanup)() {
                Ok(()) => {
                    registrations.remove(client);
                }
                Err(error) => log::warn!(
                    "[device-memory] retaining device pool client {client} after cleanup failed: {error}"
                ),
            }
        }
    }

    /// Pool retirement is forbidden while clients are live. Failed cleanup
    /// retains both the callback and its captured resources for a later retry.
    pub(crate) fn retire(&self) -> Result<bool, String> {
        let mut registrations = self.registrations.lock().unwrap();
        if registrations
            .values()
            .any(|registration| registration.users > 0)
        {
            return Ok(false);
        }
        while let Some((&client, registration)) = registrations.first_key_value() {
            (registration.cleanup)()?;
            registrations.remove(client);
        }
        Ok(true)
    }
}

pub(crate) struct PoolClientLease {
    clients: Arc<PoolClients>,
    client: &'static str,
    on_release: Option<Box<dyn FnOnce() + Send + Sync>>,
}

impl PoolClientLease {
    fn new(clients: Arc<PoolClients>, client: &'static str) -> Self {
        Self {
            clients,
            client,
            on_release: None,
        }
    }

    pub(crate) fn on_release(mut self, callback: impl FnOnce() + Send + Sync + 'static) -> Self {
        self.on_release = Some(Box::new(callback));
        self
    }
}

impl Drop for PoolClientLease {
    fn drop(&mut self) {
        self.clients.release(self.client);
        // Retirement can take the pool's initialization lock. Never invoke it
        // while holding the registration mutex used by acquisition.
        if let Some(callback) = self.on_release.take() {
            callback();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Barrier;

    #[derive(Default)]
    struct Probe {
        registered: AtomicUsize,
        cleaned: AtomicUsize,
        fail_cleanup: AtomicBool,
    }

    impl Probe {
        fn register(self: &Arc<Self>) -> Result<PoolClientCleanup, String> {
            self.registered.fetch_add(1, Ordering::SeqCst);
            let probe = self.clone();
            Ok(Box::new(move || {
                if probe.fail_cleanup.load(Ordering::SeqCst) {
                    return Err("client still owns device storage".into());
                }
                probe.cleaned.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }))
        }
    }

    #[test]
    fn shared_registration_outlives_each_model_and_does_not_retire_other_clients() {
        let clients = Arc::new(PoolClients::default());
        assert!(clients.retire().unwrap());
        let ort = Arc::new(Probe::default());
        let other = Arc::new(Probe::default());
        let model = clients.acquire("embedded-ort", || ort.register()).unwrap();
        let replica = clients.acquire("embedded-ort", || ort.register()).unwrap();
        let other_model = clients
            .acquire("other-backend", || other.register())
            .unwrap();
        assert_eq!(ort.registered.load(Ordering::SeqCst), 1);
        assert!(!clients.retire().unwrap());
        drop(model);
        assert_eq!(ort.cleaned.load(Ordering::SeqCst), 0);
        drop(replica);
        assert_eq!(ort.cleaned.load(Ordering::SeqCst), 1);
        assert_eq!(other.cleaned.load(Ordering::SeqCst), 0);
        assert!(!clients.retire().unwrap());
        drop(other_model);
        assert!(clients.retire().unwrap());
        assert_eq!(other.cleaned.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn independent_pools_and_reloads_get_independent_registrations() {
        let first = Arc::new(PoolClients::default());
        let second = Arc::new(PoolClients::default());
        let probe = Arc::new(Probe::default());
        let a = first.acquire("backend", || probe.register()).unwrap();
        let b = second.acquire("backend", || probe.register()).unwrap();
        drop(a);
        assert!(first.retire().unwrap());
        assert!(!second.retire().unwrap());
        let reloaded = first.acquire("backend", || probe.register()).unwrap();
        assert_eq!(probe.registered.load(Ordering::SeqCst), 3);
        drop((b, reloaded));
        assert_eq!(probe.cleaned.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn registration_failure_leaves_no_client_or_cleanup_callback() {
        let clients = Arc::new(PoolClients::default());
        assert!(clients
            .acquire("backend", || Err("registration failed".into()))
            .is_err());
        assert!(clients.retire().unwrap());
        let probe = Arc::new(Probe::default());
        drop(clients.acquire("backend", || probe.register()).unwrap());
        assert_eq!(probe.registered.load(Ordering::SeqCst), 1);
        assert_eq!(probe.cleaned.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn failed_cleanup_retains_resources_and_blocks_reuse_until_retry_succeeds() {
        let clients = Arc::new(PoolClients::default());
        let probe = Arc::new(Probe::default());
        probe.fail_cleanup.store(true, Ordering::SeqCst);
        let storage = Arc::new(vec![0u8; 32]);
        let retained = Arc::downgrade(&storage);
        let cleanup = probe.register().unwrap();
        let lease = clients
            .acquire("backend", || {
                Ok(Box::new(move || {
                    let _keep_storage_live = &storage;
                    cleanup()
                }))
            })
            .unwrap();
        drop(lease);
        assert!(retained.upgrade().is_some());
        assert!(clients.retire().is_err());
        assert!(clients.acquire("backend", || probe.register()).is_err());
        assert_eq!(probe.registered.load(Ordering::SeqCst), 1);
        assert!(retained.upgrade().is_some());
        probe.fail_cleanup.store(false, Ordering::SeqCst);
        let reloaded = clients.acquire("backend", || probe.register()).unwrap();
        assert!(retained.upgrade().is_none());
        assert_eq!(probe.registered.load(Ordering::SeqCst), 2);
        drop(reloaded);
        assert!(clients.retire().unwrap());
        assert_eq!(probe.cleaned.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn release_notification_can_reenter_retirement_without_holding_client_lock() {
        let clients = Arc::new(PoolClients::default());
        let probe = Arc::new(Probe::default());
        let notified = Arc::new(AtomicBool::new(false));
        let callback_clients = clients.clone();
        let callback_notified = notified.clone();
        let lease = clients
            .acquire("backend", || probe.register())
            .unwrap()
            .on_release(move || {
                assert!(callback_clients.retire().unwrap());
                callback_notified.store(true, Ordering::SeqCst);
            });
        drop(lease);
        assert!(notified.load(Ordering::SeqCst));
        assert_eq!(probe.cleaned.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn concurrent_model_preparation_shares_one_registration_until_last_release() {
        let clients = Arc::new(PoolClients::default());
        let probe = Arc::new(Probe::default());
        let loaded = Arc::new(Barrier::new(8));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let clients = clients.clone();
                let probe = probe.clone();
                let loaded = loaded.clone();
                scope.spawn(move || {
                    let lease = clients.acquire("backend", || probe.register()).unwrap();
                    loaded.wait();
                    drop(lease);
                });
            }
        });
        assert_eq!(probe.registered.load(Ordering::SeqCst), 1);
        assert_eq!(probe.cleaned.load(Ordering::SeqCst), 1);
        assert!(clients.retire().unwrap());
    }
}
