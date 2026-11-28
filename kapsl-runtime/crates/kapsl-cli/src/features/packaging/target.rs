//! Registry package references: `[<repo>/]<model>[:<label>]`.
//!
//! The repo defaults to the official public library and the label to
//! `latest`, so `kapsl pull qwen3-4b` means `kapsl pull kapsl/qwen3-4b:latest`.

/// Repo used when a reference names no repo: Kapsl Hub's official public library.
pub(crate) const DEFAULT_REPO: &str = "kapsl";
/// Label used when a reference names no label.
pub(crate) const DEFAULT_LABEL: &str = "latest";

/// Human-readable reference format used in errors and help.
pub(crate) const TARGET_FORMAT: &str = "[<repo>/]<model>[:<label>]";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelTargetRef {
    pub(crate) repo: String,
    pub(crate) model: String,
    pub(crate) label: String,
}

impl ModelTargetRef {
    pub(crate) fn as_string(&self) -> String {
        format!("{}/{}:{}", self.repo, self.model, self.label)
    }
}

pub(crate) fn is_valid_target_part(part: &str) -> bool {
    !part.is_empty()
        && part
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' || ch == '-')
}

/// Whether a raw reference relies on the default repo (it names no repo).
pub(crate) fn target_uses_default_repo(raw: &str) -> bool {
    !raw.trim().contains('/')
}

pub(crate) fn parse_model_target(raw: &str) -> Result<ModelTargetRef, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(format!(
            "Target cannot be empty. Expected format: {TARGET_FORMAT} (repo defaults to \
             `{DEFAULT_REPO}`, label to `{DEFAULT_LABEL}`)."
        ));
    }

    let (repo, rest) = trimmed.split_once('/').unwrap_or((DEFAULT_REPO, trimmed));

    if rest.contains('/') {
        return Err(format!(
            "Invalid target `{}`. Only one `/` is allowed (between repo and model).",
            trimmed
        ));
    }

    let (model, label) = rest.split_once(':').unwrap_or((rest, DEFAULT_LABEL));

    if label.contains(':') {
        return Err(format!(
            "Invalid target `{}`. Label must not contain `:`.",
            trimmed
        ));
    }

    if !is_valid_target_part(repo) {
        return Err(format!(
            "Invalid repo `{}` in target `{}`. Allowed characters: [A-Za-z0-9._-].",
            repo, trimmed
        ));
    }
    if !is_valid_target_part(model) {
        return Err(format!(
            "Invalid model `{}` in target `{}`. Allowed characters: [A-Za-z0-9._-].",
            model, trimmed
        ));
    }
    if !is_valid_target_part(label) {
        return Err(format!(
            "Invalid label `{}` in target `{}`. Allowed characters: [A-Za-z0-9._-].",
            label, trimmed
        ));
    }

    Ok(ModelTargetRef {
        repo: repo.to_string(),
        model: model.to_string(),
        label: label.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(raw: &str) -> String {
        parse_model_target(raw)
            .unwrap_or_else(|error| panic!("{raw}: {error}"))
            .as_string()
    }

    #[test]
    fn a_bare_model_uses_the_official_library_and_latest() {
        assert_eq!(parsed("qwen3-4b"), "kapsl/qwen3-4b:latest");
        assert_eq!(parsed("  qwen3-4b  "), "kapsl/qwen3-4b:latest");
        assert!(target_uses_default_repo("qwen3-4b"));
    }

    #[test]
    fn each_default_applies_on_its_own() {
        assert_eq!(
            parsed("qwen2.5-0.5b-instruct:v2"),
            "kapsl/qwen2.5-0.5b-instruct:v2"
        );
        assert_eq!(parsed("acme/gpt2"), "acme/gpt2:latest");
        assert!(!target_uses_default_repo("acme/gpt2"));
    }

    #[test]
    fn a_full_reference_is_unchanged() {
        assert_eq!(parsed("acme/gpt2:prod"), "acme/gpt2:prod");
    }

    #[test]
    fn malformed_references_are_still_rejected() {
        for raw in [
            "",
            "   ",
            "a/b/c",
            "acme/gpt2:prod:extra",
            "/gpt2",
            "acme/",
            "gpt2:",
            "acme/gpt 2",
            "bad repo/gpt2",
        ] {
            assert!(parse_model_target(raw).is_err(), "accepted {raw:?}");
        }
    }
}
