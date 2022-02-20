//! Error types

use std::error::Error;
use std::fmt;
use std::fmt::{Display, Formatter};

/// Represents an error arising from `ocl`
#[derive(Debug)]
pub struct ClgeomError{
    /// Error message
    pub message: String,

    /// Error which caused this one
    pub cause: Option<Box<dyn Error + 'static>>,
}

impl Display for ClgeomError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let result = match self.cause.as_ref() {
            Some(c) => write!(f, "{}\ncaused by\n{}", self.message, c),
            None => writeln!(f, "{}", self.message),
        };
        match result {
            Ok(_) => Ok(()), 
            Err(e) => Err(e)
        }
    }
}

impl Error for ClgeomError {}

/// Get a `ClgeomError` corresponding to the provided `ocl::Error`
///
/// # Arguments
///
/// * `T` the expected class type for the Result Ok()
/// * `e` the `ocl::Error` to look up
/// * `op` operation to report if getting the status code returns an error
///
pub fn convert_ocl_error<T>(e: &ocl::Error, op: &str) -> Result<T, ClgeomError> {
    let status_str = match e.api_status() {
        Some(s) => format!("{}", s),
        None => "[NONE]".to_owned(),
    };
    let message = e.cause().map_or_else(|| format!("OpenCL error while {}", op),|c| format!(
        "OpenCL error while {}.\nOpenCL status: {}\nCaused by:\n{}",
        op, status_str, c
    ));
    Err(ClgeomError {
        message,
        cause: None,
    })
}
