use ocl::prm::Float4;
use ocl::Buffer;

use crate::context::{ComputeContext, ParamType};
use crate::errors::ClgeomError;

pub struct TriangleMesh<'a> {
    context: &'a ComputeContext,
    data: Buffer<Float4>,
}

impl<'a> TriangleMesh<'a> {
    pub fn from_triangles(
        context: &'a ComputeContext,
        points: &'a [Float4],
    ) -> Result<TriangleMesh<'a>, ClgeomError> {
        Ok(TriangleMesh {
            context,
            data: context.create_buffer_from(points, true)?,
        })
    }

    pub fn points(&self) -> Result<Vec<[f32; 3]>, ClgeomError> {
        let buffer_content = self.context.read_buffer(&self.data)?;
        Ok(buffer_content.iter().map(|r| TriangleMesh::get_coords(r)).collect())
    }

    fn get_coords(v: &Float4) -> [f32; 3] {
        let mut result = [0.0f32; 3];
        for coord in 0..3 {
            result[coord] = match v.get(coord) {
                Some(val) => *val,
                None => 0.0f32
            };
        }
        result
    }

    pub fn scale(&self, multiplier: Float4) -> Result<(), ClgeomError> {
        let arg = vec![ParamType::Value(&multiplier)];
        self.context.execute_kernel("scale", &self.data, arg, self.data.len())
    }

    pub fn translate(&self, offset: Float4) -> Result<(), ClgeomError>{
        let arg = vec![ParamType::Value(&offset)];
        self.context.execute_kernel("translate", &self.data, arg, self.data.len())
    }
}
