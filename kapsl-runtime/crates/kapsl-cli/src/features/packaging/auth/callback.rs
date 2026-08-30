use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::time::{Duration, Instant};

pub(super) fn percent_encode_query_component(input: &str) -> String {
    let mut encoded = String::with_capacity(input.len());
    for byte in input.bytes() {
        let character = byte as char;
        if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.' | '~') {
            encoded.push(character);
        } else {
            encoded.push('%');
            encoded.push_str(&format!("{byte:02X}"));
        }
    }
    encoded
}

pub(super) fn open_browser(url: &str) -> bool {
    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg(url).status().is_ok()
    }
    #[cfg(target_os = "windows")]
    {
        Command::new("cmd")
            .args(["/C", "start", "", url])
            .status()
            .is_ok()
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        Command::new("xdg-open").arg(url).status().is_ok()
    }
}

pub(super) fn wait_for_login_callback_token(
    listener: TcpListener,
    timeout: Duration,
) -> Result<String, String> {
    let deadline = Instant::now() + timeout;
    loop {
        match listener.accept() {
            Ok((mut stream, _peer)) => return handle_login_callback_stream(&mut stream),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err("timed out waiting for login callback".to_string());
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(error) => return Err(format!("failed to accept callback connection: {error}")),
        }
    }
}

fn handle_login_callback_stream(stream: &mut TcpStream) -> Result<String, String> {
    let mut buffer = [0u8; 8192];
    let bytes_read = stream
        .read(&mut buffer)
        .map_err(|error| format!("failed to read callback request: {error}"))?;
    if bytes_read == 0 {
        return Err("empty callback request".to_string());
    }

    let request = String::from_utf8_lossy(&buffer[..bytes_read]);
    let request_line = request
        .lines()
        .next()
        .ok_or_else(|| "missing callback request line".to_string())?;
    let path = request_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| "malformed callback request line".to_string())?;

    if let Some(token) = extract_query_value_from_path(path, "token") {
        let token = token.trim();
        if !token.is_empty() {
            write_callback_response(
                stream,
                "200 OK",
                "<html><body><h3>Login complete</h3><p>You can close this tab.</p></body></html>",
            );
            return Ok(token.to_string());
        }
    }

    write_callback_response(
        stream,
        "400 Bad Request",
        "<html><body><h3>Login failed</h3><p>Token not found in callback.</p></body></html>",
    );
    Err("callback did not include token".to_string())
}

fn write_callback_response(stream: &mut TcpStream, status: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
}

fn extract_query_value_from_path(path: &str, key: &str) -> Option<String> {
    let (_, query) = path.split_once('?')?;
    query.split('&').find_map(|pair| {
        let (raw_key, raw_value) = pair.split_once('=').unwrap_or((pair, ""));
        (raw_key == key).then(|| percent_decode(raw_value))
    })
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'%' if index + 2 < bytes.len() => {
                let hex = &value[index + 1..index + 3];
                if let Ok(decoded) = u8::from_str_radix(hex, 16) {
                    output.push(decoded);
                    index += 3;
                    continue;
                }
                output.push(bytes[index]);
                index += 1;
            }
            b'+' => {
                output.push(b' ');
                index += 1;
            }
            character => {
                output.push(character);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&output).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_query_values_round_trip_reserved_characters() {
        let raw = "http://127.0.0.1:9000/callback?x=one two";
        let encoded = percent_encode_query_component(raw);
        assert_eq!(percent_decode(&encoded), raw);
    }

    #[test]
    fn callback_query_parser_decodes_token() {
        assert_eq!(
            extract_query_value_from_path("/callback?token=abc%2B123&state=x", "token"),
            Some("abc+123".to_string())
        );
    }
}
