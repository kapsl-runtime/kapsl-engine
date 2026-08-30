use kapsl_rag::extension::ConnectorRuntimeHandle;
use kapsl_rag::ConnectorClient;
use kapsl_rag_sdk::protocol::{ConnectorRequestKind, ConnectorResponseKind, ConnectorResult};
use kapsl_rag_sdk::types::SourceDescriptor;

pub(crate) fn extension_key(workspace_id: &str, extension_id: &str) -> String {
    format!("{workspace_id}:{extension_id}")
}

pub(crate) fn select_sync_source_id(
    explicit_source_id: Option<String>,
    connector_config: serde_json::Value,
    client: &mut ConnectorClient<ConnectorRuntimeHandle>,
) -> Result<String, String> {
    if let Some(source_id) = explicit_source_id {
        let trimmed = source_id.trim();
        if trimmed.is_empty() {
            return Err("source_id cannot be empty".to_string());
        }
        return Ok(trimmed.to_string());
    }

    let sources_response = client
        .request(ConnectorRequestKind::ListSources {
            config: connector_config,
        })
        .map_err(|error| format!("failed to list connector sources: {error}"))?;
    match sources_response.kind {
        ConnectorResponseKind::Err(error) => Err(error.message),
        ConnectorResponseKind::Ok(ConnectorResult::Sources(sources)) => {
            pick_default_source_id(&sources)
        }
        _ => Err("connector returned unexpected response for ListSources".to_string()),
    }
}

fn pick_default_source_id(sources: &[SourceDescriptor]) -> Result<String, String> {
    let source = sources
        .first()
        .ok_or_else(|| "connector returned no sources".to_string())?;
    let source_id = source.id.trim();
    if source_id.is_empty() {
        return Err("connector returned an empty source id".to_string());
    }
    Ok(source_id.to_string())
}

#[cfg(test)]
mod tests {
    use super::pick_default_source_id;
    use kapsl_rag_sdk::types::SourceDescriptor;
    use std::collections::HashMap;

    #[test]
    fn default_source_uses_the_first_nonempty_descriptor_id() {
        let sources = [SourceDescriptor {
            id: " source-a ".to_string(),
            name: "Source A".to_string(),
            kind: "test".to_string(),
            metadata: HashMap::new(),
        }];
        assert_eq!(
            pick_default_source_id(&sources).expect("select source"),
            "source-a"
        );
        assert!(pick_default_source_id(&[]).is_err());
    }
}
