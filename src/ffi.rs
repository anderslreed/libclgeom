//! C interface for libclgeom.h

use std::boxed::Box;
use std::ffi::{c_void, CString};
use std::mem::forget;
use std::os::raw::c_char;
use std::ptr::{null, write};

use crate::context::{ContextManager, DeviceInfo};
use crate::errors::ClgeomError;


/// Wraps a `ContextManager` for use in C.
#[repr(C)]
pub struct ClgeomContextManager {
    /// Start of the array of `ClgeomDeviceInfo` instances
    devices: *const ClgeomDeviceInfo,

    /// The wrapped `ClgeomManager`
    manager: *const c_void,

    /// The number of devices in `devices`
    n_devices: usize,
}

/// Wraps a `DeviceInfo` for use in C.
#[repr(C)]
pub struct ClgeomDeviceInfo {
    /// The id of the device
    device_id: usize,

    /// The name of the device
    device_name: *const c_char,

    /// The id of the device's platform
    platform_id: usize,

    /// The name of the device's platform
    platform_name: *const c_char,
}

/// Wraps a `ComputeContext` for use in C.
#[repr(C)]
pub struct ClgeomContext {
    /// The wrapped `ComputeContext`
    context: *const c_void,
}


/// Box the provided object and cast the raw pointer to *mut Tout
fn cast_boxed_raw<Tin, Tout>(item: Tin) -> *mut Tout {  
    Box::into_raw(Box::new(item)).cast::<Tout>()
}

// Convert a `String` to a `*const c_char`
fn string_to_c_char(s: &str) -> Result<*mut c_char, ClgeomError> {
    if let Ok(st) = CString::new(s) {
        Ok(st.into_raw())
    } else {
        Err(ClgeomError {
            message: format!("Failed to convert &str {} to *c_char", s),
            cause: None,
        })
    }
}

// Create a `ClgeomDeviceInfo` instance
fn create_c_device_info(device_info: &DeviceInfo) -> Result<ClgeomDeviceInfo, ClgeomError> {
    let device_name = match string_to_c_char(&device_info.device_name) {
        Ok(name) => name,
        Err(e) => return Err(e),
    };
    let platform_name = match string_to_c_char(&device_info.platform_name) {
        Ok(name) => name,
        Err(e) => return Err(e),
    };
    Ok(ClgeomDeviceInfo {
        device_id: device_info.device_id,
        device_name,
        platform_id: device_info.platform_id,
        platform_name,
    })
}

// Create a `ClgeomContextManager`
fn create_c_context_manager() -> Result<ClgeomContextManager, ClgeomError> {
    let manager = match ContextManager::new() {
        Ok(mgr) => mgr,
        Err(e) => return Err(e),
    };
    let device_infos = match manager.list_devices() {
        Ok(devices) => devices,
        Err(e) => return Err(e),
    };
    let c_devices_result: Result<Vec<_>, _> =
        device_infos.iter().map(create_c_device_info).collect();
    let c_devices = match c_devices_result {
        Ok(list) => list,
        Err(e) => {
            let bx = Box::new(e);
            return Err(ClgeomError {
                message: "Failed to create device info".to_owned(),
                cause: Some(bx),
            });
        }
    };
    let n_devices = c_devices.len();
    let devices = c_devices.as_ptr();
    let c_manager = ClgeomContextManager {
        devices,
        manager: cast_boxed_raw(manager),
        n_devices,
    };
    forget(c_devices);
    Ok(c_manager)
}

/// Create a `ClgeomContextManager`. Generally only one should be created per session. The `ContextManager`
/// should be deallocated using `clgeom_drop_context_manager()`. Returns a null pointer on error.
///
/// # Arguments
///
/// * `error_code`: Set to 0 if no errors are encountered, or a non-zero value to indicate an error.
///
#[no_mangle]
pub extern "C" fn clgeom_create_context_manager(
    error_code: *mut u32,
) -> ClgeomContextManager {
    let mut code = 0;
    let result = create_c_context_manager().map_or_else(
        |_| {
            code = 999;
            ClgeomContextManager {
                devices: null(),
                manager: null(),
                n_devices: 99999,
            }
        },
        |m| m,
    );
    // Safety: safe as long as error_code is a valid address
    unsafe {
        write(error_code, code);
    };
    result
}

/// Drop the specified `ClgeomContextManager` and free its memory.
///
/// # Arguments
///
/// * `mgr_ptr` a pointer to the `ClgeomContextManager` to drop.
/// * `error_code`: Set to 0 if no errors are encountered, or a non-zero value to indicate an error.
///
#[no_mangle]
pub extern "C" fn clgeom_drop_context_manager(
    mgr: ClgeomContextManager,
    error_code: *mut u32,
) {
    // Have to explicitly drop deviceinfo
    let device_ptr = mgr.devices as *mut ClgeomDeviceInfo;
    // Safety: safe as long as device_ptr and n_devices have not been altered
    unsafe { Vec::from_raw_parts(device_ptr, mgr.n_devices, mgr.n_devices) };
    // Safety: safe as long as mgr_ptr is valid
    unsafe { Box::from_raw(mgr.manager as *mut ContextManager) };
    // Safety: safe as long as error_code is valid
    unsafe { write(error_code, 0) };
}

/// Create a `ClgeomContext` with the specified device. The context should be deallocated using `clgeom_drop_context`.
///
/// # Arguments
///
/// * `mgr_ptr` a pointer to the `ClgeomContextManager` to use to create the context.
/// * `dev_ptr` a pointer to the `ClgeomDeviceInfo` to use.
/// * `error_code`: Set to 0 if no errors are encountered, or a non-zero value to indicate an error.
///
#[no_mangle]
pub extern "C" fn clgeom_create_context(
    mgr_ptr: *const ClgeomContextManager,
    dev_ptr: *const ClgeomDeviceInfo,
    error_code: *mut u32,
) -> ClgeomContext {
    // Safety: safe as long as mgr_ptr is valid
    let c_mgr = unsafe { &*mgr_ptr };
    // Safety: safe as long as mgr_ptr is valid
    let mgr = unsafe { &(*(c_mgr.manager.cast::<ContextManager>())) };
    // Safety: safe as long as mgr_ptr is valid
    let c_dev_info = unsafe { &*dev_ptr };
    let dev_info = DeviceInfo {
        device_id: c_dev_info.device_id,
        device_name: "".to_owned(),
        platform_id: c_dev_info.platform_id,
        platform_name: "".to_owned(),
    };
    let result = ClgeomContext {
        context: cast_boxed_raw(mgr.create_context(&dev_info)),
    };
    // Safety: safe as long as error_code is valid
    unsafe { write(error_code, 0) };
    result
}

/// Drop the specified `ClgeomContext` and free its memory.
///
/// # Arguments
///
/// * `context` a pointer to the `ClgeomContext` to drop.
/// * `error_code`: Set to 0 if no errors are encountered, or a non-zero value to indicate an error.
///
#[no_mangle]
pub extern "C" fn clgeom_drop_context(c_context: ClgeomContext, error_code: *mut u32) {
    // Safety: safe as long as context_ptr is valid
    unsafe {
        Box::from_raw(c_context.context as *mut c_void);
    }
    // Safety: safe as long as error_code is valid
    unsafe {
        write(error_code, 0);
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_c_context_manager() {
        let mgr = create_c_context_manager().expect("Error creating ContextManager");
        assert_ne!(mgr.n_devices, 0);
        println!("\nNumber of devices total: {}", mgr.n_devices);
    }
}
