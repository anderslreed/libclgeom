//! Core ocl tools

use std::iter::Iterator;

use ocl::flags::{MemFlags, DEVICE_TYPE_GPU};
use ocl::prm::Float4;
use ocl::traits::OclPrm;
use ocl::{Buffer, Context, Device, Kernel, Platform, Queue};

use crate::compile::get_program;
use crate::errors::{rewrap_ocl_result, ClgeomError};

/// Represents an `ocl::Device`. Only valid for the `ContextManager` which created it.
pub struct DeviceInfo {
    /// Unique identifier among devices on a platform.
    pub device_id: usize,

    /// The name of the device.
    pub device_name: String,

    /// id of the device's platform.
    pub platform_id: usize,

    /// The name of the platform.
    pub platform_name: String,
}

type BufferResult<T> = Result<Buffer<T>, ClgeomError>;

/// Wraps a `ocl::ComputeContext`.
pub struct ComputeContext {
    /// The wrapped `ocl::Context`.
    context: Context,
    queue: Queue,
}

impl ComputeContext {
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

    pub fn execute_kernel(
        &self,
        name: &str,
        data: &Buffer<Float4>,
        args: Vec<ParamType>,
        size: usize,
    ) -> Result<(), ClgeomError> {
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

    pub fn read_buffer(&self, buffer: &Buffer<Float4>) -> Result<Vec<Float4>, ClgeomError>{
        let mut result = vec![Float4::new(0.0, 0.0, 0.0, 0.0); buffer.len()];
        rewrap_ocl_result(buffer.read(&mut result).queue(&self.queue).enq(), "reading result")?;
        Ok(result)
    }
}

pub enum ParamType<'a> {
    Buffer(&'a Buffer<Float4>),
    Value(&'a Float4),
}

/// An `ocl::Platform` and its vector of `ocl::Device`.
struct PlatformDevices {
    /// Devices for this platform
    devices: Vec<Device>,

    /// The platform
    platform: Platform,
}

impl PlatformDevices {
    /// Wrap `ocl::Platform` and its `ocl::Device` instances
    fn new(platform: Platform, devices: Vec<Device>) -> Self {
        Self { devices, platform }
    }
}

/// Factory class for `ComputeContext`.
/// Holds available platforms and devices.
pub struct ContextManager {
    /// Available `OpenCL` platforms with their devices.
    ocl_platforms: Vec<PlatformDevices>,
}

impl ContextManager {
    /// Create a new `ContextManager` instance.
    pub fn new() -> Result<Self, ClgeomError> {
        let raw_platforms = Platform::list();
        let platform_devices: Result<Vec<_>, _> =
            raw_platforms.iter().map(|p| unwrap_devices(*p)).collect();
        let ocl_platforms = match platform_devices {
            Ok(platforms) => platforms,
            Err(e) => return Err(e),
        };
        Ok(Self { ocl_platforms })
    }

    /// Get `DeviceInfo` for all `OpenCL` devices.
    pub fn list_devices(&self) -> Result<Vec<DeviceInfo>, ClgeomError> {
        let devices_result: Result<Vec<_>, _> = self
            .ocl_platforms
            .iter()
            .enumerate()
            .map(|p| create_device_infos(p.0, p.1))
            .collect();
        let devices = match devices_result {
            Ok(device) => device,
            Err(e) => return Err(e),
        };
        Ok(devices.into_iter().flatten().collect())
    }

    /// Create a `ComputeContext` with the indicated device.
    ///
    /// # Arguments
    ///
    /// * `device` - device to create context with.
    ///
    pub fn create_context(&self, device: &DeviceInfo) -> Result<ComputeContext, ClgeomError> {
        let mut builder = Context::builder();
        let ocl_platform = self.ocl_platforms.get(device.platform_id).ok_or_else(|| {
            ClgeomError::new(&format!("Error getting platform {}", device.platform_id))
        })?;
        let ocl_device = ocl_platform.devices.get(device.device_id).ok_or_else(|| {
            ClgeomError::new(&format!(
                "getting device {} for platform {}",
                device.device_id, device.platform_id
            ))
        })?;
        builder.platform(ocl_platform.platform);
        builder.devices(ocl_device);
        let context = rewrap_ocl_result(builder.build(), "creating context")?;
        let queue = rewrap_ocl_result(
            Queue::new(&context, *ocl_device, None),
            "creating command queue",
        )?;
        Ok(ComputeContext { context, queue })
    }
}

// Get a list of devices for the specified platform
fn unwrap_devices(platform: Platform) -> Result<PlatformDevices, ClgeomError> {
    let devices = rewrap_ocl_result(
        Device::list(platform, Some(DEVICE_TYPE_GPU)),
        "listing devices",
    )?;
    Ok(PlatformDevices::new(platform, devices))
}

// Create `DeviceInfo` vector for a platform
fn create_device_infos(
    platform_id: usize,
    platform_devices: &PlatformDevices,
) -> Result<Vec<DeviceInfo>, ClgeomError> {
    let name = rewrap_ocl_result(
        platform_devices.platform.name(),
        "creating DeviceInfo instances",
    )?;
    platform_devices
        .devices
        .iter()
        .enumerate()
        .map(|d| create_device_info(platform_id, name.clone(), d.0, *d.1))
        .collect()
}

// Create a `DeviceInfo` for the given `ocl::Device`.
fn create_device_info(
    platform_id: usize,
    platform_name: String,
    device_id: usize,
    device: Device,
) -> Result<DeviceInfo, ClgeomError> {
    let device_name = rewrap_ocl_result(device.name(), "getting device name")?;
    Ok(DeviceInfo {
        device_id,
        device_name,
        platform_id,
        platform_name,
    })
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
        let devices = mgr.list_devices().expect("Error listing devices");
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
        let device_info = &mgr.list_devices().expect("Error listing devices")[0];
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
        cxt
            .execute_kernel(
                "translate",
                &buffer_a,
                vec![ParamType::Value(&data_b)],
                data_a.len(),
            )
            .unwrap();

        let result = cxt.read_buffer(buffer_a).unwrap();

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
