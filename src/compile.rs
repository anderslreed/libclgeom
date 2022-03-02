//! Build `OpenCL` programs from source

use ocl::{Context, Device, Program};

use crate::errors::{rewrap_ocl_result, ClgeomError};

// Library of source code strings generated from files in src/opencl/
macro_rules! get_source {
    ($fn_name:ident) => {{
        use crate::errors::ClgeomError;
        match $fn_name {
            "translate" => Ok(format!(
                include_str!("opencl/fn_inpl_scalar.c"),
                name = "translate",
                op = "+=",
                T1 = "float4",
                T2 = "float4"
            )),
            "scale" => Ok(format!(
                include_str!("opencl/fn_inpl_scalar.c"),
                name = "scale",
                op = "*=",
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
pub fn get_program(function: &str, context: &Context, target: &Device) -> Result<Program, ClgeomError> {
    rewrap_ocl_result(
        Program::builder()
            .source(get_source!(function)?)
            .devices(target)
            .build(context),
        "building OpenCL program",
    )
}

#[cfg(test)]
mod tests {

    #[test]
    fn get_translate() {
        let name = "translate";
        let expected = concat!(
            "__kernel void translate(__global float4 *A, const float4 B) {\n",
            "    int i = get_global_id(0);\n",
            "    A[i] += B;\n",
            "}"
        );
        assert_eq!(get_source!(name).unwrap(), expected);
    }
}
