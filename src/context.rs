//! Core ocl tools

use std::iter::Iterator;

use ocl::{flags::DEVICE_TYPE_GPU, Context, Device, Platform};

use crate::errors::{convert_ocl_error, ClgeomError};

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

/// Wraps a `ocl::ComputeContext`.
pub struct ComputeContext {
    /// The wrapped `ocl::Context`.
    _context: Context,
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
        let ocl_platform = match self.ocl_platforms.get(device.platform_id) {
            Some(pfm) => pfm,
            None => {
                return Err(ClgeomError::new(format!(
                    "Error getting platform {}",
                    device.platform_id
                )))
            }
        };
        let ocl_device = match ocl_platform.devices.get(device.device_id) {
            Some(dev) => dev,
            None => {
                return Err(ClgeomError::new(format!(
                    "Error getting device {} for platform {}",
                    device.device_id, device.platform_id
                )))
            }
        };
        builder.platform(ocl_platform.platform);
        builder.devices(ocl_device);
        let context = match builder.build() {
            Ok(ctx) => ctx,
            Err(e) => return Err(ClgeomError::new(format!("Failed to create context. {:?}", e))),
        };
        Ok(ComputeContext { _context: context })
    }
}

// Get a list of devices for the specified platform
fn unwrap_devices(platform: Platform) -> Result<PlatformDevices, ClgeomError> {
    match Device::list(platform, Some(DEVICE_TYPE_GPU)) {
        Ok(devices) => Ok(PlatformDevices::new(platform, devices)),
        Err(e) => convert_ocl_error(&e, "reading OpenCL error status while listing devices"),
    }
}

// Create `DeviceInfo` vector for a platform
fn create_device_infos(
    platform_id: usize,
    platform_devices: &PlatformDevices,
) -> Result<Vec<DeviceInfo>, ClgeomError> {
    let name = match platform_devices.platform.name() {
        Ok(val) => val,
        Err(e) => return convert_ocl_error(&e, "creating DeviceInfo instances"),
    };
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
    let device_name = match device.name() {
        Ok(txt) => txt,
        Err(e) => return convert_ocl_error(&e, "getting device name"),
    };
    Ok(DeviceInfo {
        device_id,
        device_name,
        platform_id,
        platform_name,
    })
}

/*
    TESTS
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_context_manager() {
        let mgr = ContextManager::new().expect("Error creating ContextManager");
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
}
