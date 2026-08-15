//! OpenAI-compatible HTTP surface.
//!
//! These routes exist so an existing OpenAI client can be pointed at a Kapsl
//! runtime by changing its base URL and nothing else. They are a thin
//! translation layer: every request ends up on the same `ReplicaPool` paths as
//! `/api/models/:id/infer`, with the same admission, runtime-pressure shedding
//! and cancellation behaviour.

use super::*;

mod chat;
mod types;

use types::openai_error;

pub(crate) struct OpenAiRoutesConfig {
    pub(crate) models: Arc<ModelManager>,
    pub(crate) inference: Arc<InferenceService>,
    pub(crate) log_sensitive_ids: bool,
}

/// A model that is loaded, has a live pool, and can serve inference.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedModel {
    pub(crate) id: u32,
    /// The name echoed back to the client, so `model` in the response matches
    /// what `/v1/models` advertises rather than whatever alias was sent.
    pub(crate) name: String,
}

/// Resolve the OpenAI `model` field to a loaded model.
///
/// OpenAI clients address models by name; the runtime addresses them by a u32
/// id. Accepted forms, in order: the exact advertised name, `name:version`,
/// and a bare numeric id (which is what `/api/models/:id/infer` takes, and is
/// convenient when scripting against both surfaces).
pub(crate) fn resolve_model(
    models: &ModelManager,
    requested: &str,
) -> Result<ResolvedModel, String> {
    let served = served_models(models);
    resolve_model_from(&served, requested)
}

/// The models that are both registered and backed by a live pool. Anything
/// else cannot serve a request, so it is not addressable and not advertised.
fn served_models(models: &ModelManager) -> Vec<ModelInfo> {
    models
        .registry()
        .list()
        .into_iter()
        .filter(|model| models.contains_pool(model.id))
        .collect()
}

fn resolve_model_from(served: &[ModelInfo], requested: &str) -> Result<ResolvedModel, String> {
    let requested = requested.trim();
    if requested.is_empty() {
        return Err("model is required".to_string());
    }

    // Primary replicas first: replicas share a name, and the primary is the
    // one whose id the rest of the API treats as canonical.
    let by_preference = |a: &ModelInfo, b: &ModelInfo| {
        let primary = |model: &ModelInfo| model.id == model.base_model_id;
        primary(b).cmp(&primary(a)).then(a.id.cmp(&b.id))
    };

    let pick = |mut candidates: Vec<ModelInfo>| -> Option<ResolvedModel> {
        candidates.sort_by(by_preference);
        candidates.into_iter().next().map(|model| ResolvedModel {
            id: model.id,
            name: model.name,
        })
    };

    let by_name: Vec<ModelInfo> = served
        .iter()
        .filter(|model| model.name.eq_ignore_ascii_case(requested))
        .cloned()
        .collect();
    if let Some(resolved) = pick(by_name) {
        return Ok(resolved);
    }

    if let Some((name, version)) = requested.rsplit_once(':') {
        let by_name_version: Vec<ModelInfo> = served
            .iter()
            .filter(|model| {
                model.name.eq_ignore_ascii_case(name) && model.version.eq_ignore_ascii_case(version)
            })
            .cloned()
            .collect();
        if let Some(resolved) = pick(by_name_version) {
            return Ok(resolved);
        }
    }

    if let Ok(id) = requested.parse::<u32>() {
        if let Some(model) = served.iter().find(|model| model.id == id) {
            return Ok(ResolvedModel {
                id: model.id,
                name: model.name.clone(),
            });
        }
    }

    // Name the alternatives: a bare "not found" leaves the caller guessing at
    // a string they cannot see from the client side.
    let mut available: Vec<String> = served.iter().map(|model| model.name.clone()).collect();
    available.sort();
    available.dedup();
    if available.is_empty() {
        return Err(format!(
            "The model '{requested}' does not exist: no models are currently loaded"
        ));
    }
    Err(format!(
        "The model '{requested}' does not exist. Available models: {}",
        available.join(", ")
    ))
}

fn model_object(model: &ModelInfo) -> serde_json::Value {
    serde_json::json!({
        "id": model.name,
        "object": "model",
        "created": model.loaded_at,
        "owned_by": "kapsl",
    })
}

pub(crate) fn build_openai_routes(
    config: OpenAiRoutesConfig,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let OpenAiRoutesConfig {
        models,
        inference,
        log_sensitive_ids,
    } = config;

    // GET /v1/models
    let models_for_list = models.clone();
    let list_models = warp::path!("v1" / "models").and(warp::get()).map(move || {
        let mut served = served_models(&models_for_list);
        // Only advertise primaries: replicas duplicate the name and are an
        // internal scaling detail.
        served.retain(|model| model.id == model.base_model_id);
        served.sort_by(|a, b| a.name.cmp(&b.name));
        let data: Vec<serde_json::Value> = served.iter().map(model_object).collect();
        reply_into_response(warp::reply::json(&serde_json::json!({
            "object": "list",
            "data": data,
        })))
    });

    // GET /v1/models/:model
    let models_for_get = models.clone();
    let get_model =
        warp::path!("v1" / "models" / String)
            .and(warp::get())
            .map(
                move |requested: String| match resolve_model(&models_for_get, &requested) {
                    Ok(resolved) => match models_for_get.registry().get(resolved.id) {
                        Some(model) => {
                            reply_into_response(warp::reply::json(&model_object(&model)))
                        }
                        None => openai_error(
                            warp::http::StatusCode::NOT_FOUND,
                            format!("The model '{requested}' does not exist"),
                            "invalid_request_error",
                        ),
                    },
                    Err(message) => openai_error(
                        warp::http::StatusCode::NOT_FOUND,
                        message,
                        "invalid_request_error",
                    ),
                },
            );

    let chat_completions = chat::build_chat_completions_route(chat::ChatCompletionsConfig {
        models,
        inference,
        log_sensitive_ids,
    });

    list_models
        .or(get_model)
        .unify()
        .or(chat_completions)
        .unify()
        .map(reply_into_response)
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(id: u32, name: &str, version: &str) -> ModelInfo {
        ModelInfo::new(
            id,
            name.to_string(),
            version.to_string(),
            "gguf".to_string(),
            "CPU".to_string(),
            "none".to_string(),
            format!("/models/{name}.aimod"),
        )
    }

    fn replica(id: u32, base_model_id: u32, name: &str) -> ModelInfo {
        ModelInfo::new_replica(
            id,
            id - base_model_id,
            base_model_id,
            name.to_string(),
            "1".to_string(),
            "gguf".to_string(),
            "CPU".to_string(),
            "none".to_string(),
            format!("/models/{name}.aimod"),
        )
    }

    #[test]
    fn resolves_by_advertised_name() {
        let served = vec![model(7, "qwen2.5-7b-instruct", "1")];
        let resolved = resolve_model_from(&served, "qwen2.5-7b-instruct").expect("should resolve");
        assert_eq!(resolved.id, 7);
        assert_eq!(resolved.name, "qwen2.5-7b-instruct");
    }

    #[test]
    fn resolves_case_insensitively_and_ignores_surrounding_space() {
        let served = vec![model(7, "Qwen2.5-7B-Instruct", "1")];
        assert_eq!(
            resolve_model_from(&served, "  qwen2.5-7b-instruct ")
                .expect("should resolve")
                .id,
            7
        );
    }

    #[test]
    fn resolves_name_and_version() {
        let served = vec![model(1, "gemma", "2"), model(2, "gemma", "3")];
        assert_eq!(
            resolve_model_from(&served, "gemma:3")
                .expect("should resolve")
                .id,
            2
        );
    }

    #[test]
    fn resolves_a_bare_numeric_id() {
        let served = vec![model(42, "gemma", "3")];
        let resolved = resolve_model_from(&served, "42").expect("should resolve");
        assert_eq!(resolved.id, 42);
        // The advertised name is echoed back, not the numeric alias.
        assert_eq!(resolved.name, "gemma");
    }

    #[test]
    fn prefers_the_primary_over_its_replicas() {
        // Replicas share a name; the primary is the canonical id.
        let served = vec![replica(9, 3, "gemma"), model(3, "gemma", "1")];
        assert_eq!(
            resolve_model_from(&served, "gemma")
                .expect("should resolve")
                .id,
            3
        );
    }

    #[test]
    fn a_name_matching_nothing_lists_the_alternatives() {
        let served = vec![model(1, "gemma", "1"), model(2, "qwen", "1")];
        let error = resolve_model_from(&served, "gpt-4o").expect_err("should not resolve");
        assert!(error.contains("gpt-4o"), "{error}");
        assert!(error.contains("gemma") && error.contains("qwen"), "{error}");
    }

    #[test]
    fn an_empty_runtime_says_so_rather_than_listing_nothing() {
        let error = resolve_model_from(&[], "gemma").expect_err("should not resolve");
        assert!(error.contains("no models are currently loaded"), "{error}");
    }

    #[test]
    fn an_empty_model_field_is_rejected() {
        let served = vec![model(1, "gemma", "1")];
        assert!(resolve_model_from(&served, "   ").is_err());
    }
}
