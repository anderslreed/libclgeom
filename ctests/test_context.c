#ifndef _TEST_CONTEXT_CC
#define _TEST_CONTEXT_CC

#include "stdio.h"

#include "../target/debug/libclgeom.h"
#include <stdint.h>
#include <stdlib.h>

// Check exit code and reset for the next test
#define CHECK_ERROR(var)                                                       \
  if (var) {                                                                   \
    return var;                                                                \
  } else {                                                                     \
    var = 100;                                                                 \
  }

int test_context() {
  printf("test_context.c\n");
  uint32_t err = 100; // Each call should set a value

  // Get context manager
  ClgeomContextManager manager = clgeom_create_context_manager(&err);
  CHECK_ERROR(err)

  printf("\tNumber of devices: %lu\n", manager.n_devices);
  // First device on first platform
  ClgeomDeviceInfo default_device = manager.devices[0];
  printf("\tDefault device: %s\n", default_device.device_name);

  // Create context
  printf("Get context\n");
  ClgeomContext cxt = clgeom_create_context(&manager, &default_device, &err);
  CHECK_ERROR(err)

  // Clean up
  printf("Drop context\n");
  clgeom_drop_context(cxt, &err);
  CHECK_ERROR(err)

  printf("Drop context manager\n");
  clgeom_drop_context_manager(manager, &err);
  CHECK_ERROR(err)

  return 0;
}

#endif