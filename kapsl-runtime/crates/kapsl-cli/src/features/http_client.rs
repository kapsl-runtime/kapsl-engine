//! Shared outbound HTTP client configuration.
//!
//! This module is intentionally independent from individual CLI features so
//! packaging, extensions, providers, and runtime support can use the same TLS
//! and error-handling policy without depending on one another.

/// Formats an outbound HTTP error without exposing `ureq` details to callers.
pub(crate) fn format_remote_http_error(error: ureq::Error) -> String {
    match error {
        ureq::Error::StatusCode(status) => format!("Remote backend returned HTTP {status}"),
        other => other.to_string(),
    }
}

fn build_native_tls_http_agent(timeout: Option<std::time::Duration>) -> ureq::Agent {
    let mut config = ureq::Agent::config_builder().tls_config(
        ureq::tls::TlsConfig::builder()
            .provider(ureq::tls::TlsProvider::NativeTls)
            // Ureq's WebPki default disables Schannel's trusted roots on Windows.
            .root_certs(ureq::tls::RootCerts::PlatformVerifier)
            .build(),
    );
    if let Some(timeout) = timeout {
        config = config
            .timeout_global(Some(timeout))
            .timeout_per_call(Some(timeout));
    }
    config.build().into()
}

/// Builds a native-TLS agent using the operating system's trusted roots.
pub(crate) fn native_tls_http_agent() -> ureq::Agent {
    build_native_tls_http_agent(None)
}

/// Builds a native-TLS agent with one timeout applied globally and per call.
pub(crate) fn native_tls_http_agent_with_timeout(timeout: std::time::Duration) -> ureq::Agent {
    build_native_tls_http_agent(Some(timeout))
}

/// Builds an unbounded transfer agent suitable for large uploads/downloads.
///
/// Rustls avoids macOS native-TLS rejecting certificates with long validity
/// periods while the absent global timeout permits long artifact transfers.
pub(crate) fn http_agent_for_transfer() -> ureq::Agent {
    ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::Rustls)
                .build(),
        )
        .timeout_global(None)
        .build()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_tls_http_agent_uses_platform_roots() {
        let agent = native_tls_http_agent();
        let tls_config = agent.config().tls_config();

        assert_eq!(tls_config.provider(), ureq::tls::TlsProvider::NativeTls);
        assert!(matches!(
            tls_config.root_certs(),
            ureq::tls::RootCerts::PlatformVerifier
        ));
    }
}
