use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};

const INDEX_HTML: &str = include_str!("index.html");

pub async fn index() -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        INDEX_HTML,
    )
        .into_response()
}
