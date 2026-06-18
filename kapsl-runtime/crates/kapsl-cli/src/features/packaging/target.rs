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

pub(crate) fn parse_model_target(raw: &str) -> Result<ModelTargetRef, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(
            "Target cannot be empty. Expected format: <repo_name>/<model>:<label>.".to_string(),
        );
    }

    let (repo, rest) = trimmed.split_once('/').ok_or_else(|| {
        format!(
            "Invalid target `{}`. Expected format: <repo_name>/<model>:<label>.",
            trimmed
        )
    })?;

    if rest.contains('/') {
        return Err(format!(
            "Invalid target `{}`. Only one `/` is allowed (between repo and model).",
            trimmed
        ));
    }

    let (model, label) = rest.split_once(':').ok_or_else(|| {
        format!(
            "Invalid target `{}`. Expected format: <repo_name>/<model>:<label>.",
            trimmed
        )
    })?;

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
