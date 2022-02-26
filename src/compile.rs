//! Build `OpenCL` programs from source

use ocl::builders::ContextBuilder;
use ocl::{Device, Program};

use crate::errors::{rewrap_ocl_result, ClgeomError};

// Library of source code strings generated from files in src/opencl/
macro_rules! get_source {
    ($fn_name:ident) => {{
        use crate::errors::ClgeomError;
        match $fn_name {
            "add" => Ok(format!(
                include_str!("opencl/fn_inpl.c"),
                name = "add",
                op = "+=",
                T1 = "float4",
                T2 = "float4"
            )),
            "sub" => Ok(format!(
                include_str!("opencl/fn_inpl.c"),
                name = "sub",
                op = "-=",
                T1 = "float4",
                T2 = "float4"
            )),
            "mul" => Ok(format!(
                include_str!("opencl/fn_inpl.c"),
                name = "mul",
                op = "*=",
                T1 = "float4",
                T2 = "float4"
            )),
            "div" => Ok(format!(
                include_str!("opencl/fn_inpl.c"),
                name = "div",
                op = "/=",
                T1 = "float4",
                T2 = "float4"
            )),
            &_ => Err(ClgeomError::new(&format!("Unknown function: {}", $fn_name))),
        }
    }};
}

/// Return a compiled `ocl::Program` for the specified function
///
/// # Arguments
///
/// * `function` - name of function to retrieve program for
/// * `target` - device which will run the program
///
pub fn get_program(function: &str, target: Device) -> Result<Program, ClgeomError> {
    let context = rewrap_ocl_result(
        ContextBuilder::new().devices(target).build(),
        "creating context",
    )?;
    rewrap_ocl_result(
        Program::builder()
            .source(get_source!(function)?)
            .devices(target)
            .build(&context),
        "building OpenCL program",
    )
}

#[cfg(test)]
mod tests {

    #[test]
    fn get_add() {
        let name = "add";
        let expected = concat!(
            "__kernel void add(__global float4 *A, __global const float4 *B) {\n",
            "    int i = get_global_id(0);\n",
            "    A[i] += B[i];\n",
            "}"
        );
        assert_eq!(get_source!(name).unwrap(), expected);
    }
}
