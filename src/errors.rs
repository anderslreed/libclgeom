//! Error types

use std::error::Error;
use std::fmt;
use std::fmt::{Display, Formatter};

/// Represents an error arising from `ocl`
#[derive(Debug)]
pub struct ClgeomError {
    /// Error message
    pub message: String,

    /// Error which caused this one
    pub cause: Option<Box<dyn Error>>,
}

impl Display for ClgeomError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let result = match self.cause.as_ref() {
            Some(c) => write!(f, "{}\ncaused by\n{}", self.message, c),
            None => writeln!(f, "{}", self.message),
        };
        match result {
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }
}

impl Error for ClgeomError {}

/// Get a `ClgeomError` corresponding to the provided `std::error::Error`
///
/// # Arguments
///
/// * `T` the expected class type for the Result Ok()
/// * `e` the `std::error::Error` to look up
/// * `op` operation to report if getting the status code returns an error
///
pub fn convert_std_error<T>(e: Box<dyn Error>, op: &str) -> Result<T, ClgeomError> {
    let message = format!("Error while {}.", op);
    Err(ClgeomError {
        message,
        cause: Some(e),
    })
}

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
    // ocl::Error.cause() returns ocl::error::Fail, which is private, preventing recursion.
    // Just add the error message from the cause, if any, to the output.
    let cause_str = match e.cause() {
        Some(c) => format!("Caused by:\n{}", c),
        None => "".to_owned(),
    };
    let message = format!(
        "OpenCL error while {}.\nOpenCL status: {}\n{}",
        op, status_str, cause_str
    );
    Err(ClgeomError {
        message,
        cause: None,
    })
}
