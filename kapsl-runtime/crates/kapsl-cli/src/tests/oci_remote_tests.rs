use super::*;

#[test]
fn test_parse_model_target_valid_and_invalid() {
    assert_eq!(
        parse_model_target("alice/mnist:prod")
            .expect("valid target")
            .as_string(),
        "alice/mnist:prod"
    );
    assert!(parse_model_target("alice/mnist").is_err());
    assert!(parse_model_target("alice/mnist:").is_err());
    assert!(parse_model_target("/mnist:prod").is_err());
    assert!(parse_model_target("alice/mnist:pro:d").is_err());
}

#[test]
fn test_parse_oci_remote_prefix_basic_and_trailing_slash() {
    assert_eq!(
        parse_oci_remote_prefix("oci://ghcr.io").expect("valid prefix"),
        "ghcr.io"
    );
    assert_eq!(
        parse_oci_remote_prefix("oci://ghcr.io/acme/").expect("valid prefix"),
        "ghcr.io/acme"
    );
}

#[test]
fn test_parse_oci_remote_prefix_rejects_tag_and_digest() {
    assert!(parse_oci_remote_prefix("oci://ghcr.io/acme:latest").is_err());
    assert!(parse_oci_remote_prefix("oci://ghcr.io/acme@sha256:0123").is_err());
}

#[test]
fn test_build_oci_repo_for_target() {
    let target = parse_model_target("alice/mnist:prod").expect("target");
    assert_eq!(
        build_oci_repo_for_target("oci://ghcr.io", &target).expect("repo"),
        "ghcr.io/alice/mnist"
    );
    assert_eq!(
        build_oci_repo_for_target("oci://ghcr.io/team", &target).expect("repo"),
        "ghcr.io/team/alice/mnist"
    );
}

#[test]
fn test_build_oci_reference_tag_and_digest_override() {
    let repo = "ghcr.io/acme/kapsl";
    let tag = "prod";
    assert_eq!(
        build_oci_reference(repo, tag, None).expect("tag ref"),
        "ghcr.io/acme/kapsl:prod"
    );

    let digest = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    assert_eq!(
        build_oci_reference(repo, tag, Some(digest)).expect("digest ref"),
        format!("ghcr.io/acme/kapsl@{}", digest)
    );
    assert_eq!(
        build_oci_reference(repo, tag, Some(&format!("@{}", digest))).expect("digest ref"),
        format!("ghcr.io/acme/kapsl@{}", digest)
    );
    assert_eq!(
        build_oci_reference(repo, tag, Some(&format!("{}@{}", repo, digest))).expect("full ref"),
        format!("{}@{}", repo, digest)
    );
}

#[test]
fn test_parse_oras_manifest_digest_prefers_digest_line() {
    let digest = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let output = format!(
        "Uploading...\nUploaded layer: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nDigest: {}\nDone\n",
        digest
    );
    assert_eq!(
        parse_oras_manifest_digest(&output),
        Some(digest.to_string())
    );
}
