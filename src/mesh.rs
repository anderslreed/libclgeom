//! Triangle mesh struct and associated operations

use ocl::prm::Float4;
use ocl::Buffer;

use crate::context::{ComputeContext, ParamType};
use crate::errors::ClgeomError;

/// A mesh of triangles defining a surface
pub struct TriangleMesh<'a> {
    context: &'a ComputeContext,
    data: Buffer<Float4>,
}

impl<'a> TriangleMesh<'a> {
    /// Create a `TriangleMesh` from a array containing point triples for triangles, with duplicate
    /// points
    /// 
    /// # Arguments
    /// 
    /// * `context` - the context 
    pub fn from_list(
        &self,
        points: &'a [Float4],
    ) -> Result<TriangleMesh<'a>, ClgeomError> {
        Ok(TriangleMesh {
            context: &self.context,
            data: self.context.create_buffer_from(points, true)?,
        })
    }

    /// Return a vector of point triples representing the mesh
    pub fn triangles(&self) -> Result<Vec<[[f32; 3]; 3]>, ClgeomError> {
        let buffer_content = self.context.read_buffer(&self.data)?;
        if (buffer_content.len() % 3) != 0 {
            return Err(ClgeomError::new("Buffer length is not a multiple of 3."))
        } 
        let mut acc = TriangleAccumulator::new(buffer_content.len());
        for pt in buffer_content {
            acc = acc.add_point(&pt);
        }
        Ok(acc.result)
    }

    // Get a raw array of components for vector v
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

    /// Scale the mesh around the origin
    /// 
    /// * `multuplier` - the amount to scale each coordinate by
    /// 
    pub fn scale(&self, multiplier: Float4) -> Result<(), ClgeomError> {
        let arg = vec![ParamType::Value(&multiplier)];
        self.context.execute_kernel("scale", &self.data, arg)
    }

    /// Translate the mesh
    /// 
    /// * `offest` - the movement vector
    /// 
    pub fn translate(&self, offset: Float4) -> Result<(), ClgeomError>{
        let arg = vec![ParamType::Value(&offset)];
        self.context.execute_kernel("translate", &self.data, arg)
    }
}

// Helps convert array of points to vector of point triples
struct TriangleAccumulator {
    pub result: Vec<[[f32; 3]; 3]>,
    next_pt_index: usize,
    tmp_triangle: [[f32; 3]; 3],
}

impl TriangleAccumulator {
    // Create a new accumulator with the specified expected result size
    fn new(triangle_count: usize) -> TriangleAccumulator{
        TriangleAccumulator {
            result: Vec::with_capacity(triangle_count / 3),
            tmp_triangle: [[0.0f32; 3]; 3],
            next_pt_index: 0
        }
    }

    // Add a point to the accumulator's result
    fn add_point(mut self, pt: &Float4) -> TriangleAccumulator {
        self.tmp_triangle[self.next_pt_index] = TriangleMesh::get_coords(pt);
        self.next_pt_index += 1;
        if self.next_pt_index == 3 {
            self.result.push(self.tmp_triangle.clone());
            self.next_pt_index = 0;
        }
        self
    }
}
