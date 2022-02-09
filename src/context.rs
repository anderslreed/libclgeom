use ocl::{Context, Device, flags::DEVICE_TYPE_GPU, Platform};

pub struct DeviceInfo {
    device: Device,
    platform: Platform
}

impl DeviceInfo {
    pub fn name(&self) -> String { 
        let name = match self.device.name() {
            Ok(txt) => txt,
            Err(e) => panic!("Failed to get device name. {:?}", e)
        };
        name
    }
}

pub type PlatformDevices = Vec<Device>;

pub struct ComputeContext {
    context: Context
}

pub struct ContextManager {
    platform_devices: Vec<PlatformDevices>
}

impl ContextManager {
    pub fn new() -> ContextManager {
        ContextManager { platform_devices: get_all_devices() }
    }

    // Get OpenCL platform info
    pub fn list_platforms(&self) -> Vec<PlatformDevices> {
        self.platform_devices.to_vec()
    }

    // Create a ComputeContext with the indicated device
    pub fn get_context(&self, device: &DeviceInfo) -> ComputeContext {
        let mut builder = Context::builder();
        builder.platform(device.platform);
        builder.devices(device.device);
        let context = match builder.build() {
            Ok(ctx) => ctx,
            Err(e) => panic!("Failed to create context. {:?}", e)
        };
        ComputeContext { context }
    }    
}

fn accumulate_platforms(mut acc: Vec<PlatformDevices>, pfm: &Platform) -> Vec<PlatformDevices> {
    let devices = match Device::list(pfm, Some(DEVICE_TYPE_GPU)) {
        Ok(devs) => devs,
        Err(e) => panic!("Failed to get devices. {:?}", e)
    };
    if !devices.is_empty() {
        acc.push(devices);
    }
    acc
}

fn get_all_devices() -> Vec<PlatformDevices> {
    let platforms = Platform::list();
    assert_ne!(platforms.len(), 0, "No platforms found.");

    let platform_devices = platforms.iter().fold(Vec::new(),  accumulate_platforms);
    assert_ne!(platform_devices.len(), 0, "No devices found for any platform.");
    platform_devices
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{stdout, Write};

    #[test]
    fn get_context_manager() {
        let mgr = ContextManager::new();
        let devices = mgr.list_platforms();
        assert!(!devices.is_empty());
        let name = match devices[0][0].name() {
            Ok(txt) => txt,
            Err(e) => panic!("Failed to get device name. {:?}", e)
        };
        stdout().write_all(format!("Default device: {:?}\n", name).as_bytes()).unwrap();
    }
}
