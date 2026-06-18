use super::*;

#[derive(RustEmbed)]
#[folder = "../../ui"]
struct UiAssets;

pub(crate) fn build_static_routes() -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    // Static file serving for web UI (from embedded assets)
    let index_route = warp::path::end().and(warp::get()).and_then(|| async {
        if let Some(content) = UiAssets::get("index.html") {
            Ok::<_, warp::Rejection>(
                warp::http::Response::builder()
                    .header("content-type", "text/html; charset=utf-8")
                    .header("cache-control", "no-cache")
                    .body(content.data.into_owned())
                    .unwrap(),
            )
        } else {
            Err(warp::reject::not_found())
        }
    });

    let ui_static_files = warp::path("ui")
        .and(warp::path::tail())
        .and(warp::get())
        .and_then(|tail: warp::path::Tail| async move {
            let filename = tail.as_str();
            if let Some(content) = UiAssets::get(filename) {
                let mime_type = mime_guess::from_path(filename)
                    .first_or_octet_stream()
                    .to_string();
                Ok::<_, warp::Rejection>(
                    warp::http::Response::builder()
                        .header("content-type", mime_type)
                        .header("cache-control", "no-cache")
                        .body(content.data.into_owned())
                        .unwrap(),
                )
            } else {
                Err(warp::reject::not_found())
            }
        });

    let static_files = ui_static_files;

    index_route
        .or(static_files)
        .map(reply_into_response)
        .boxed()
}
