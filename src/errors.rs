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

impl ClgeomError {
    /// Create a new `ClgeomError`
    /// 
    /// # Arguments
    /// 
    /// * `message` the error message
    pub fn new(message: &str) -> Self {
        Self {
            message: message.to_owned(),
            cause: None,
        }
    }
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

/// If the provided result is an `ocl::error::Error`, return a corresponding `ClgeomError`. Otherwise
/// reurn the value of the result
///
/// # Arguments
///
/// * `T` the expected class type for the Result Ok()
/// * `r` the `ocl::error::Result` to unwrap
/// * `op` operation to report if getting the status code returns an error
///
pub fn rewrap_ocl_result<T>(r: ocl::error::Result<T>, op: &str) -> Result<T, ClgeomError> {
    match r {
        Ok(val) => Ok(val),
        Err(e) => {
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
                "OpenCL error while {}.\nOpenCL status: {}\nError message: {}\nCaused by: {}\n",
                op, status_str, cause_str, e
            );
            Err(ClgeomError {
                message,
                cause: None,
            })
        }    
    }
    
}

#[cfg(test)]
mod tests {
    use ocl::{Context, Device, Platform, Program};
    use ocl::flags::DEVICE_TYPE_GPU;
    use super::*;

    #[test]
    fn report_ocl_error() {
        let expected = concat!(
            "OpenCL error while building OpenCL program.\n",
            "OpenCL status: [NONE]\n",
            "Error message: \n",
            "Caused by: \n\n",
            "###################### OPENCL PROGRAM BUILD DEBUG OUTPUT ######################\n\n",
            "<kernel>:3:1: error: unknown type name 'invalid'\n",
            "invalid source causes an error\n",
            "^\n",
            "<kernel>:3:9: error: variable has address space that is not supported in program scope declaration\n",
            "invalid source causes an error\n",
            "        ^\n",
            "<kernel>:3:15: error: expected ';' after top level declarator\n",
            "invalid source causes an error\n",
            "              ^\n              ",
            ";\n\u{0}\n",
            "###############################################################################\n\n\n"
        );

        let target = Device::list(Platform::default(), Some(DEVICE_TYPE_GPU)).unwrap()[0];
        let context = Context::builder().devices(target).build().unwrap();
        let error = match rewrap_ocl_result(
            Program::builder()
                .source("invalid source causes an error")
                .devices(target)
                .build(&context),
            "building OpenCL program",
        ) {
            Ok(_) => panic!("Error expected"),
            Err(e) => e
        };

        println!("{}", error.message);
        assert_eq!(error.message, expected);
    }
}
