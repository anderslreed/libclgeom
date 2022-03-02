//! Core ocl tools

use std::iter::Iterator;

use ocl::enums::{DeviceInfo as OclDeviceInfo, DeviceInfoResult};
use ocl::flags::{MemFlags, DEVICE_TYPE_GPU};
use ocl::prm::Float4;
use ocl::traits::OclPrm;
use ocl::{Buffer, Context, Device, Kernel, Platform, Queue};

use crate::compile::get_program;
use crate::errors::{rewrap_ocl_result, ClgeomError};

/// Represents an `ocl::Device`. Only valid for the `ContextManager` which created it.
#[derive(Clone)]
pub struct DeviceInfo {
    /// OpenCL device
    pub device: Device,

    /// The name of the device.
    pub device_name: String,

    /// The name of the platform.
    pub platform_name: String,
}

impl DeviceInfo {
    /// Create a `DeviceInfo` for the given `ocl::Device`.
    ///
    /// # Arguments
    ///
    /// * `device` - the `ocl::Device` to use
    ///
    pub fn from_device(device: Device) -> Result<DeviceInfo, ClgeomError> {
        let device_name = rewrap_ocl_result(device.name(), "getting device name")?;
        let platform = rewrap_ocl_result(
            device.info(OclDeviceInfo::Platform),
            &format!("getting platform for device {}", device_name),
        )?;
        let platform_name = match platform {
            DeviceInfoResult::Platform(p) => {
                rewrap_ocl_result(Platform::from(p).name(), "getting platform name")
            }
            _ => {
                return Err(ClgeomError {
                    message: "Error getting platform.".to_owned(),
                    cause: None,
                })
            }
        }?;
        Ok(DeviceInfo {
            device,
            device_name,
            platform_name,
        })
    }
}

type BufferResult<T> = Result<Buffer<T>, ClgeomError>;

/// Wraps a `ocl::ComputeContext`.
pub struct ComputeContext {
    /// The wrapped `ocl::Context`.
    context: Context,
    queue: Queue,
}

impl ComputeContext {
    /// Create an `ocl::Buffer` from an array
    ///
    /// # Arguments
    ///
    /// * `T` - the type of the input elements
    /// * `data` - the values to initialize the buffer with
    /// * `allow_write` - true to create a read/write buffer
    ///
    pub fn create_buffer_from<T: OclPrm>(&self, data: &[T], allow_write: bool) -> BufferResult<T> {
        let flags = if allow_write {
            MemFlags::READ_WRITE
        } else {
            MemFlags::READ_ONLY
        };
        rewrap_ocl_result(
            Buffer::builder()
                .context(&self.context)
                .copy_host_slice(data)
                .flags(flags)
                .len(data.len())
                .build(),
            "creating buffer",
        )
    }

    /// Create an empty `ocl::Buffer`
    ///
    /// # Arguments
    ///
    /// * `T` - the type of the elements the buffer is to hold
    /// * `size` - the number of elements to allocate the buffer for
    ///
    pub fn create_empty_buffer<T: OclPrm>(&self, size: usize) -> BufferResult<T> {
        rewrap_ocl_result(
            Buffer::builder()
                .context(&self.context)
                .flags(MemFlags::READ_WRITE)
                .len(size)
                .build(),
            "creating empty buffer",
        )
    }

    /// Execute a named kernel
    ///
    /// # Arguments
    ///
    /// * `name` - the name of the kernel to run
    /// * `data` - the `ocl::Buffer` containing data to execute the kernel on
    /// * `args` - additional arguments
    ///
    pub fn execute_kernel(
        &self,
        name: &str,
        data: &Buffer<Float4>,
        args: Vec<ParamType>,
    ) -> Result<(), ClgeomError> {
        let size = data.len();
        let devices = self.context.devices();
        let device = match devices.get(0) {
            Some(v) => v,
            None => return Err(ClgeomError::new("Error getting device")),
        };
        let program = get_program(name, &self.context, device)?;
        let mut kernel_builder = Kernel::builder();
        kernel_builder.arg(data);
        for arg in args {
            match arg {
                ParamType::Buffer(content) => {
                    kernel_builder.arg(content);
                }
                ParamType::Value(content) => {
                    kernel_builder.arg(content);
                }
            };
        }
        let kernel = rewrap_ocl_result(
            kernel_builder
                .global_work_size(size)
                .name(name)
                .program(&program)
                .queue(self.queue.clone())
                .build(),
            &format!("building kernel for function: {}", name),
        )?;
        // Safety: user is responsible for supplying appropriate kernel args
        unsafe { rewrap_ocl_result(kernel.enq(), &format!("running kernel: {}", name)) }
    }

    /// Return the data in the provided `ocl::Buffer`
    ///
    /// # Arguments
    ///
    /// * `buffer` - the buffer to read
    ///
    pub fn read_buffer(&self, buffer: &Buffer<Float4>) -> Result<Vec<Float4>, ClgeomError> {
        let mut result = vec![Float4::new(0.0, 0.0, 0.0, 0.0); buffer.len()];
        rewrap_ocl_result(
            buffer.read(&mut result).queue(&self.queue).enq(),
            "reading result",
        )?;
        Ok(result)
    }
}

/// An arbitrary input parameter for a kernel
pub enum ParamType<'a> {
    // An `ocl::Buffer`
    Buffer(&'a Buffer<Float4>),
    // A single value
    Value(&'a Float4),
}

/// Factory class for `ComputeContext`.
/// Holds available platforms and devices.
pub struct ContextManager {
    /// Available `OpenCL` devices.
    ocl_devices: Vec<DeviceInfo>,
}

impl ContextManager {
    /// Create a new `ContextManager` instance.
    pub fn new() -> Result<Self, ClgeomError> {
        let raw_platforms = Platform::list();
        let mut ocl_devices = Vec::new();
        for platform in raw_platforms {
            let mut tmp_devices = wrap_ocl_devices(&platform)?;
            ocl_devices.append(&mut tmp_devices);
        }
        Ok(Self { ocl_devices })
    }

    /// Get `DeviceInfo` for all `OpenCL` devices.
    pub fn list_devices(&self) -> Vec<DeviceInfo> {
        self.ocl_devices.clone()
    }

    /// Create a `ComputeContext` with the indicated device.
    ///
    /// # Arguments
    ///
    /// * `device` - device to create context with.
    ///
    pub fn create_context(&self, device: &DeviceInfo) -> Result<ComputeContext, ClgeomError> {
        let mut builder = Context::builder();
        builder.devices(device.device);
        let context = rewrap_ocl_result(builder.build(), "creating context")?;
        let queue = rewrap_ocl_result(
            Queue::new(&context, device.device, None),
            "creating command queue",
        )?;
        Ok(ComputeContext { context, queue })
    }
}

// Get a list of DeviceInfo instances for the specified platform
fn wrap_ocl_devices(platform: &Platform) -> Result<Vec<DeviceInfo>, ClgeomError> {
    let raw_devices = rewrap_ocl_result(
        Device::list(platform, Some(DEVICE_TYPE_GPU)),
        "listing devices",
    )?;
    raw_devices
        .into_iter()
        .map(|d| DeviceInfo::from_device(d))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_context_manager() -> ContextManager {
        ContextManager::new().expect("Error creating ContextManager")
    }

    #[test]
    fn get_context_manager() {
        let mgr = create_context_manager();
        let devices = mgr.list_devices();
        assert!(!devices.is_empty());

        println!("\nNumber of devices: {}", devices.len());
        println!(
            "Default device: {}",
            devices
                .get(0)
                .expect("Error getting defailt device")
                .device_name
        );
    }

    #[test]
    fn translate() {
        let mgr = create_context_manager();
        let device_info = &mgr.list_devices()[0];
        let cxt = mgr.create_context(&device_info).unwrap();

        let data_a: &[Float4; 2] = &[
            Float4::new(1.0, 2.0, 3.0, 0.0),
            Float4::new(4.1, 5.2, 6.3, 0.0),
        ];
        let data_b = Float4::new(0.1, 0.2, 0.3, 0.0);
        let expected: &[Float4; 2] = &[
            Float4::new(1.1, 2.2, 3.3, 0.0),
            Float4::new(4.2, 5.4, 6.6, 0.0),
        ];

        let buffer_a: Buffer<Float4> = cxt.create_buffer_from(data_a, true).unwrap();
        cxt.execute_kernel("translate", &buffer_a, vec![ParamType::Value(&data_b)])
            .unwrap();

        let result = cxt.read_buffer(&buffer_a).unwrap();

        for i in 0..data_a.len() {
            for j in 0..3 {
                let exp = expected[i].get(j).unwrap();
                let dat = result[i].get(j).unwrap();
                println!("{},{}: {}", i, j, exp);
                println!("{},{}: {}", i, j, dat);
                assert!((exp - dat).abs() < 0.000001);
            }
        }
    }
}
