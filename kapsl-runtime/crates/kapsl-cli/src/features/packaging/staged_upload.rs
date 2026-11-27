//! Staged, content-scanned uploads to a Kapsl Hub registry.
//!
//! Kapsl Hub publishes a pushed package only after scanning it. A client opts
//! in by POSTing `{"content_scan": true, "size_bytes": n}` to
//! `<artifact>/upload`. The Hub answers with an `upload_id` naming an isolated
//! staging object plus either one presigned URL or, for large packages, a
//! multipart layout of presigned part URLs. Once the bytes are stored, the
//! client POSTs `<artifact>/confirm`; the Hub scans the staged bytes and only
//! then publishes them under the artifact reference.
//!
//! A registry without object storage answers the opt-in with 404 or 501, and
//! the caller falls back to a direct PUT. A registry that predates staging may
//! answer with a bare `upload_url` and no `upload_id`; such an upload needs no
//! confirmation.
//!
//! The bearer token is sent only to the registry's control endpoints, never to
//! the presigned object-storage URLs.
//!
//! This module depends only on `ureq`, `serde`, `serde_json` and `std`.

use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Attempts per object-storage PUT. Only transport errors and 5xx responses
/// are retried; a 4xx means the signed request itself is wrong.
const STORAGE_PUT_ATTEMPTS: usize = 3;

/// Error from a staged upload step, keeping the HTTP status for callers that
/// react to it (for example, re-authenticating after a 401).
#[derive(Debug)]
pub(crate) struct StagedUploadError {
    pub(crate) status_code: Option<u16>,
    pub(crate) message: String,
}

impl StagedUploadError {
    fn local(message: impl Into<String>) -> Self {
        Self {
            status_code: None,
            message: message.into(),
        }
    }

    fn from_ureq(context: impl std::fmt::Display, error: ureq::Error) -> Self {
        let status_code = match &error {
            ureq::Error::StatusCode(code) => Some(*code),
            _ => None,
        };
        Self {
            status_code,
            message: format!("{context}: {error}"),
        }
    }
}

/// One presigned request that stores bytes in object storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StorageTarget {
    pub(crate) url: String,
    pub(crate) method: String,
    pub(crate) headers: Vec<(String, String)>,
}

/// How the registry asked for the package bytes to be delivered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UploadPlan {
    /// One request carrying the whole package. `upload_id` is `None` only for
    /// registries that predate staged uploads; those need no confirmation.
    Single {
        target: StorageTarget,
        upload_id: Option<String>,
    },
    /// Consecutive parts of `part_size` bytes (the last may be shorter).
    Multipart {
        upload_id: String,
        multipart_upload_id: String,
        part_size: u64,
        parts: Vec<StorageTarget>,
    },
}

#[derive(Deserialize)]
struct UploadResponse {
    upload_url: Option<String>,
    method: Option<String>,
    headers: Option<HashMap<String, String>>,
    upload_id: Option<String>,
    multipart: Option<MultipartLayout>,
}

#[derive(Deserialize)]
struct MultipartLayout {
    upload_id: String,
    part_size_bytes: u64,
    parts: Vec<MultipartPart>,
}

#[derive(Deserialize)]
struct MultipartPart {
    part_number: u64,
    upload_url: String,
    method: Option<String>,
    headers: Option<HashMap<String, String>>,
}

fn storage_target(
    url: String,
    method: Option<String>,
    headers: Option<HashMap<String, String>>,
) -> Result<StorageTarget, String> {
    if url.trim().is_empty() {
        return Err("upload response has an empty upload URL".to_string());
    }
    let method = method
        .map(|value| value.trim().to_ascii_uppercase())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "PUT".to_string());
    if method != "PUT" && method != "POST" {
        return Err(format!(
            "upload response asks for unsupported method {method}"
        ));
    }
    let mut headers: Vec<(String, String)> = headers.unwrap_or_default().into_iter().collect();
    headers.sort();
    Ok(StorageTarget {
        url,
        method,
        headers,
    })
}

/// Parses and validates the registry's answer to the upload opt-in for a
/// package of `size_bytes` bytes.
pub(crate) fn parse_upload_plan(body: &str, size_bytes: u64) -> Result<UploadPlan, String> {
    let response: UploadResponse =
        serde_json::from_str(body).map_err(|error| format!("invalid upload response: {error}"))?;
    let upload_id = response
        .upload_id
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    if let Some(layout) = response.multipart {
        let upload_id = upload_id
            .ok_or_else(|| "multipart upload response is missing upload_id".to_string())?;
        if layout.upload_id.trim().is_empty() {
            return Err("multipart upload response is missing multipart.upload_id".to_string());
        }
        let part_size = layout.part_size_bytes;
        if part_size == 0 {
            return Err("multipart upload response has a zero part size".to_string());
        }
        let expected_parts = size_bytes.div_ceil(part_size);
        let mut parts = layout.parts;
        parts.sort_by_key(|part| part.part_number);
        if parts.len() as u64 != expected_parts
            || parts
                .iter()
                .enumerate()
                .any(|(index, part)| part.part_number != index as u64 + 1)
        {
            return Err(format!(
                "multipart upload response lists {} parts; a {size_bytes}-byte package in \
                 {part_size}-byte parts needs parts 1..={expected_parts}",
                parts.len()
            ));
        }
        let parts = parts
            .into_iter()
            .map(|part| storage_target(part.upload_url, part.method, part.headers))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(UploadPlan::Multipart {
            upload_id,
            multipart_upload_id: layout.upload_id,
            part_size,
            parts,
        });
    }

    let url = response
        .upload_url
        .ok_or_else(|| "upload response is missing upload_url".to_string())?;
    Ok(UploadPlan::Single {
        target: storage_target(url, response.method, response.headers)?,
        upload_id,
    })
}

/// Byte range `(offset, length)` of zero-based part `index`.
pub(crate) fn part_range(index: usize, part_size: u64, size_bytes: u64) -> (u64, u64) {
    let offset = index as u64 * part_size;
    (offset, part_size.min(size_bytes.saturating_sub(offset)))
}

fn with_authorization<B>(
    request: ureq::RequestBuilder<B>,
    authorization_header: Option<&str>,
) -> ureq::RequestBuilder<B> {
    match authorization_header {
        Some(header) => request.header("Authorization", header),
        None => request,
    }
}

fn post_json(
    agent: &ureq::Agent,
    url: &str,
    authorization_header: Option<&str>,
    body: &serde_json::Value,
) -> Result<ureq::http::Response<ureq::Body>, ureq::Error> {
    let request = agent.post(url).header("Content-Type", "application/json");
    with_authorization(request, authorization_header).send(body.to_string().as_bytes())
}

/// Opts in to a staged upload. Returns `None` when the registry does not offer
/// staged uploads (HTTP 404 or 501), in which case the caller should PUT the
/// package directly.
pub(crate) fn request_upload_plan(
    agent: &ureq::Agent,
    artifact_url: &str,
    authorization_header: Option<&str>,
    size_bytes: u64,
) -> Result<Option<UploadPlan>, StagedUploadError> {
    let endpoint = format!("{artifact_url}/upload");
    let body = serde_json::json!({ "content_scan": true, "size_bytes": size_bytes });
    let mut response = match post_json(agent, &endpoint, authorization_header, &body) {
        Ok(response) => response,
        Err(ureq::Error::StatusCode(404 | 501)) => return Ok(None),
        Err(error) => {
            return Err(StagedUploadError::from_ureq(
                format!("Failed to request an upload URL from {endpoint}"),
                error,
            ))
        }
    };
    let text = response.body_mut().read_to_string().map_err(|error| {
        StagedUploadError::local(format!(
            "Failed to read upload response from {endpoint}: {error}"
        ))
    })?;
    parse_upload_plan(&text, size_bytes)
        .map(Some)
        .map_err(|error| StagedUploadError::local(format!("{endpoint}: {error}")))
}

fn has_header(headers: &[(String, String)], name: &str) -> bool {
    headers
        .iter()
        .any(|(key, _)| key.eq_ignore_ascii_case(name))
}

/// Streams `length` bytes of `source_path` starting at `offset` to one
/// presigned target, retrying transport errors and 5xx responses.
fn put_to_storage(
    agent: &ureq::Agent,
    target: &StorageTarget,
    source_path: &Path,
    offset: u64,
    length: u64,
) -> Result<(), StagedUploadError> {
    let mut last_error = None;
    for _ in 0..STORAGE_PUT_ATTEMPTS {
        let mut file = File::open(source_path).map_err(|error| {
            StagedUploadError::local(format!(
                "Failed to open {} for upload: {error}",
                source_path.display()
            ))
        })?;
        file.seek(SeekFrom::Start(offset)).map_err(|error| {
            StagedUploadError::local(format!(
                "Failed to seek {} to byte {offset}: {error}",
                source_path.display()
            ))
        })?;
        let mut reader = file.take(length);
        let mut request = if target.method == "POST" {
            agent.post(&target.url)
        } else {
            agent.put(&target.url)
        };
        // An explicit Content-Length makes ureq send a length-delimited body;
        // object stores reject chunked presigned uploads.
        request = request.header("Content-Length", length.to_string());
        if !has_header(&target.headers, "Content-Type") {
            request = request.header("Content-Type", "application/octet-stream");
        }
        for (key, value) in &target.headers {
            request = request.header(key.as_str(), value.as_str());
        }
        match request.send(ureq::SendBody::from_reader(&mut reader)) {
            Ok(_) => return Ok(()),
            Err(ureq::Error::StatusCode(code)) if code < 500 => {
                return Err(StagedUploadError {
                    status_code: Some(code),
                    message: format!("Object storage rejected the upload with HTTP {code}"),
                });
            }
            Err(error) => last_error = Some(error),
        }
    }
    let error = last_error.expect("at least one attempt ran");
    Err(StagedUploadError::from_ureq(
        format!("Failed to upload to object storage after {STORAGE_PUT_ATTEMPTS} attempts"),
        error,
    ))
}

fn confirm_upload(
    agent: &ureq::Agent,
    artifact_url: &str,
    authorization_header: Option<&str>,
    body: &serde_json::Value,
) -> Result<(), StagedUploadError> {
    let endpoint = format!("{artifact_url}/confirm");
    post_json(agent, &endpoint, authorization_header, body)
        .map(|_| ())
        .map_err(|error| {
            StagedUploadError::from_ureq(
                format!("The registry did not publish the uploaded package ({endpoint})"),
                error,
            )
        })
}

/// Delivers the package bytes as `plan` describes and asks the registry to
/// scan and publish them. A failed multipart upload is aborted so the registry
/// can discard the stored parts.
pub(crate) fn execute_upload_plan(
    agent: &ureq::Agent,
    artifact_url: &str,
    authorization_header: Option<&str>,
    source_path: &Path,
    size_bytes: u64,
    plan: &UploadPlan,
) -> Result<(), StagedUploadError> {
    match plan {
        UploadPlan::Single { target, upload_id } => {
            put_to_storage(agent, target, source_path, 0, size_bytes)?;
            match upload_id {
                Some(upload_id) => confirm_upload(
                    agent,
                    artifact_url,
                    authorization_header,
                    &serde_json::json!({
                        "upload_id": upload_id,
                        "expected_size_bytes": size_bytes,
                    }),
                ),
                None => Ok(()),
            }
        }
        UploadPlan::Multipart {
            upload_id,
            multipart_upload_id,
            part_size,
            parts,
        } => {
            for (index, target) in parts.iter().enumerate() {
                let (offset, length) = part_range(index, *part_size, size_bytes);
                if let Err(error) = put_to_storage(agent, target, source_path, offset, length) {
                    // Best effort: the upload already failed, and an abort that
                    // also fails leaves only an unreachable staged object.
                    let _ = confirm_upload(
                        agent,
                        artifact_url,
                        authorization_header,
                        &serde_json::json!({
                            "upload_id": upload_id,
                            "multipart_upload_id": multipart_upload_id,
                            "abort": true,
                        }),
                    );
                    return Err(StagedUploadError {
                        status_code: error.status_code,
                        message: format!(
                            "Part {} of {}: {}",
                            index + 1,
                            parts.len(),
                            error.message
                        ),
                    });
                }
            }
            confirm_upload(
                agent,
                artifact_url,
                authorization_header,
                &serde_json::json!({
                    "upload_id": upload_id,
                    "multipart_upload_id": multipart_upload_id,
                    "expected_size_bytes": size_bytes,
                }),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};
    use std::thread::JoinHandle;

    #[derive(Debug, Clone)]
    struct Recorded {
        method: String,
        path: String,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    }

    impl Recorded {
        fn header(&self, name: &str) -> Option<&str> {
            self.headers
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case(name))
                .map(|(_, value)| value.as_str())
        }

        fn json(&self) -> serde_json::Value {
            serde_json::from_slice(&self.body).expect("request body is JSON")
        }
    }

    /// A one-connection-per-request HTTP/1.1 server that records every request
    /// and answers with `respond(request, base_url) -> (status, body)`.
    struct MockServer {
        base_url: String,
        recorded: Arc<Mutex<Vec<Recorded>>>,
        handle: Option<JoinHandle<()>>,
    }

    impl MockServer {
        fn start(
            requests: usize,
            respond: impl Fn(&Recorded, &str) -> (u16, String) + Send + 'static,
        ) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
            let base_url = format!("http://{}", listener.local_addr().expect("local addr"));
            let recorded = Arc::new(Mutex::new(Vec::new()));
            let thread_recorded = recorded.clone();
            let thread_base = base_url.clone();
            let handle = std::thread::spawn(move || {
                for _ in 0..requests {
                    let (stream, _) = listener.accept().expect("accept");
                    let request = read_request(&stream);
                    let (status, body) = respond(&request, &thread_base);
                    thread_recorded.lock().expect("record").push(request);
                    let mut stream = stream;
                    write!(
                        stream,
                        "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .expect("write response");
                }
            });
            Self {
                base_url,
                recorded,
                handle: Some(handle),
            }
        }

        fn finish(mut self) -> Vec<Recorded> {
            self.handle
                .take()
                .expect("handle")
                .join()
                .expect("mock server thread");
            self.recorded.lock().expect("recorded").clone()
        }
    }

    fn read_request(stream: &TcpStream) -> Recorded {
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).expect("request line");
        let mut parts = line.split_whitespace();
        let method = parts.next().expect("method").to_string();
        let path = parts.next().expect("path").to_string();
        let mut headers = Vec::new();
        loop {
            let mut header = String::new();
            reader.read_line(&mut header).expect("header line");
            let header = header.trim_end();
            if header.is_empty() {
                break;
            }
            let (key, value) = header.split_once(':').expect("header separator");
            headers.push((key.trim().to_string(), value.trim().to_string()));
        }
        let length = headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
            .map(|(_, value)| value.parse::<usize>().expect("content length"))
            .unwrap_or(0);
        let mut body = vec![0; length];
        reader.read_exact(&mut body).expect("request body");
        Recorded {
            method,
            path,
            headers,
            body,
        }
    }

    fn agent() -> ureq::Agent {
        ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(20)))
            .build()
            .into()
    }

    fn package(bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("model.aimod");
        std::fs::write(&path, bytes).expect("write package");
        (dir, path)
    }

    /// Runs the full opt-in, storage and confirmation sequence the way
    /// `HttpArtifactTransport::upload` does.
    fn push(server: &MockServer, path: &Path) -> Result<Option<UploadPlan>, StagedUploadError> {
        let artifact_url = format!("{}/v1/aimod/team/model:latest", server.base_url);
        let size = std::fs::metadata(path).expect("metadata").len();
        let agent = agent();
        let plan = request_upload_plan(&agent, &artifact_url, Some("Bearer secret"), size)?;
        if let Some(plan) = &plan {
            execute_upload_plan(
                &agent,
                &artifact_url,
                Some("Bearer secret"),
                path,
                size,
                plan,
            )?;
        }
        Ok(plan)
    }

    #[test]
    fn parses_a_single_staged_upload() {
        let plan = parse_upload_plan(
            r#"{"status":"ok","upload_url":"https://store/x?sig=1","method":"PUT",
                "expires_at":"2026-01-01T00:00:00Z","headers":{"x-ms-blob-type":"BlockBlob"},
                "upload_id":"u-1"}"#,
            10,
        )
        .expect("parse");
        assert_eq!(
            plan,
            UploadPlan::Single {
                target: StorageTarget {
                    url: "https://store/x?sig=1".to_string(),
                    method: "PUT".to_string(),
                    headers: vec![("x-ms-blob-type".to_string(), "BlockBlob".to_string())],
                },
                upload_id: Some("u-1".to_string()),
            }
        );
    }

    #[test]
    fn a_legacy_presigned_response_needs_no_confirmation() {
        let plan = parse_upload_plan(r#"{"upload_url":"https://store/x","headers":null}"#, 10)
            .expect("parse");
        assert!(
            matches!(plan, UploadPlan::Single { upload_id: None, ref target } if target.method == "PUT")
        );
    }

    #[test]
    fn rejects_inconsistent_multipart_layouts() {
        let layout = |parts: &str, part_size: u64| {
            format!(
                r#"{{"upload_id":"u","multipart":{{"upload_id":"m","part_size_bytes":{part_size},
                    "expires_at":"2026-01-01T00:00:00Z","parts":[{parts}]}}}}"#
            )
        };
        let part = |n: u64| {
            format!(r#"{{"part_number":{n},"upload_url":"https://store/{n}","method":"PUT"}}"#)
        };
        let ok = layout(&[part(2), part(1), part(3)].join(","), 4);
        match parse_upload_plan(&ok, 10).expect("valid layout") {
            UploadPlan::Multipart {
                parts, part_size, ..
            } => {
                assert_eq!(part_size, 4);
                let urls: Vec<_> = parts.iter().map(|p| p.url.as_str()).collect();
                assert_eq!(
                    urls,
                    ["https://store/1", "https://store/2", "https://store/3"]
                );
            }
            other => panic!("expected multipart, got {other:?}"),
        }
        let missing = layout(&[part(1), part(2)].join(","), 4);
        assert!(parse_upload_plan(&missing, 10).is_err());
        let gap = layout(&[part(1), part(2), part(4)].join(","), 4);
        assert!(parse_upload_plan(&gap, 10).is_err());
        assert!(parse_upload_plan(&layout(&part(1), 0), 10).is_err());
        let no_upload_id = ok.replacen(r#""upload_id":"u","#, "", 1);
        assert!(parse_upload_plan(&no_upload_id, 10).is_err());
    }

    #[test]
    fn part_ranges_cover_the_package_exactly() {
        assert_eq!(part_range(0, 4, 10), (0, 4));
        assert_eq!(part_range(1, 4, 10), (4, 4));
        assert_eq!(part_range(2, 4, 10), (8, 2));
    }

    #[test]
    fn single_upload_stores_bytes_without_the_token_and_then_confirms() {
        let payload = b"aimod package bytes".to_vec();
        let (_dir, path) = package(&payload);
        let server = MockServer::start(3, |request, base| match request.path.as_str() {
            "/v1/aimod/team/model:latest/upload" => (
                200,
                format!(
                    r#"{{"status":"ok","upload_url":"{base}/staging/u-1?sig=abc","method":"PUT","headers":{{}},"upload_id":"u-1"}}"#
                ),
            ),
            "/staging/u-1?sig=abc" => (200, String::new()),
            "/v1/aimod/team/model:latest/confirm" => (
                200,
                r#"{"status":"ok","reference":"team/model:latest","size_bytes":19}"#.to_string(),
            ),
            other => (404, format!("unexpected {other}")),
        });
        let plan = push(&server, &path).expect("staged push");
        assert!(matches!(plan, Some(UploadPlan::Single { .. })));
        let requests = server.finish();

        let opt_in = &requests[0];
        assert_eq!(opt_in.method, "POST");
        assert_eq!(opt_in.header("Authorization"), Some("Bearer secret"));
        assert_eq!(
            opt_in.json(),
            serde_json::json!({ "content_scan": true, "size_bytes": payload.len() })
        );

        let stored = &requests[1];
        assert_eq!(stored.method, "PUT");
        assert_eq!(stored.body, payload);
        assert_eq!(stored.header("Content-Length"), Some("19"));
        assert_eq!(stored.header("Transfer-Encoding"), None);
        assert_eq!(
            stored.header("Authorization"),
            None,
            "token leaked to storage"
        );

        let confirm = &requests[2];
        assert_eq!(confirm.header("Authorization"), Some("Bearer secret"));
        assert_eq!(
            confirm.json(),
            serde_json::json!({ "upload_id": "u-1", "expected_size_bytes": payload.len() })
        );
    }

    #[test]
    fn multipart_upload_sends_each_slice_then_confirms_with_both_ids() {
        let payload = b"0123456789".to_vec();
        let (_dir, path) = package(&payload);
        let server = MockServer::start(5, |request, base| {
            if request.path.ends_with("/upload") {
                let parts: Vec<String> = (1..=3)
                    .map(|n| {
                        format!(
                            r#"{{"part_number":{n},"upload_url":"{base}/part/{n}","method":"PUT"}}"#
                        )
                    })
                    .collect();
                (
                    200,
                    format!(
                        r#"{{"status":"ok","upload_id":"u-2","multipart":{{"upload_id":"m-2","part_size_bytes":4,"expires_at":"2026-01-01T00:00:00Z","parts":[{}]}}}}"#,
                        parts.join(",")
                    ),
                )
            } else {
                (200, "{}".to_string())
            }
        });
        push(&server, &path).expect("multipart push");
        let requests = server.finish();

        let slices: Vec<(&str, &[u8])> = requests[1..4]
            .iter()
            .map(|r| (r.path.as_str(), r.body.as_slice()))
            .collect();
        assert_eq!(
            slices,
            [
                ("/part/1", &b"0123"[..]),
                ("/part/2", &b"4567"[..]),
                ("/part/3", &b"89"[..]),
            ]
        );
        assert!(requests[1..4]
            .iter()
            .all(|r| r.header("Authorization").is_none()));
        assert_eq!(
            requests[4].json(),
            serde_json::json!({
                "upload_id": "u-2",
                "multipart_upload_id": "m-2",
                "expected_size_bytes": 10,
            })
        );
    }

    #[test]
    fn a_registry_without_staged_uploads_falls_back() {
        let (_dir, path) = package(b"bytes");
        let server = MockServer::start(1, |_, _| (501, "not configured".to_string()));
        assert_eq!(push(&server, &path).expect("fallback"), None);
        server.finish();
    }

    #[test]
    fn a_rejected_part_aborts_the_multipart_upload() {
        let (_dir, path) = package(b"0123456789");
        let server = MockServer::start(3, |request, base| {
            if request.path.ends_with("/upload") {
                (
                    200,
                    format!(
                        r#"{{"upload_id":"u-3","multipart":{{"upload_id":"m-3","part_size_bytes":8,"expires_at":"x","parts":[
                            {{"part_number":1,"upload_url":"{base}/part/1"}},{{"part_number":2,"upload_url":"{base}/part/2"}}]}}}}"#
                    ),
                )
            } else if request.path == "/part/1" {
                (403, "SignatureDoesNotMatch".to_string())
            } else {
                (204, String::new())
            }
        });
        let error = push(&server, &path).expect_err("rejected part");
        assert_eq!(error.status_code, Some(403));
        assert!(error.message.contains("Part 1 of 2"), "{}", error.message);
        let requests = server.finish();
        assert_eq!(requests[1].path, "/part/1", "a 4xx is not retried");
        assert_eq!(
            requests[2].json(),
            serde_json::json!({ "upload_id": "u-3", "multipart_upload_id": "m-3", "abort": true })
        );
    }

    #[test]
    fn a_rejected_confirmation_reports_the_scan_result() {
        let (_dir, path) = package(b"bytes");
        let server = MockServer::start(3, |request, base| {
            if request.path.ends_with("/upload") {
                (
                    200,
                    format!(r#"{{"upload_url":"{base}/staging","upload_id":"u-4"}}"#),
                )
            } else if request.path.ends_with("/confirm") {
                (422, "artifact rejected by content scan".to_string())
            } else {
                (200, String::new())
            }
        });
        let error = push(&server, &path).expect_err("rejected by scan");
        assert_eq!(error.status_code, Some(422));
        server.finish();
    }
}
