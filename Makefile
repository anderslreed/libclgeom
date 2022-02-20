CC ?= /bin/cc

C_SRC_PATH := ctests
C_FILES := $(shell find $(C_SRC_PATH)/ -type f -name '*.c')
RUST_SRC_PATH := src
RUST_FILES := $(shell find $(RUST_SRC_PATH)/ -type f -name '*.rs')
FFI_MODS := context
FFI_FILES := $(patsubst %,  src/%.rs, $(FFI_MODS))
TARGET_PATH := target

.PHONY: all build header release test

all: test

build: $(RUST_FILES) header
	cargo build

header: $(TARGET_PATH)/debug/libclgeom.h

release: $(RUST_FILES) $(TARGET_PATH)/release/libclgeom.h
	cargo build

target/debug/libclgeom.h: $(FFI_FILES)
	cbindgen -o $(TARGET_PATH)/debug/libclgeom.h
	# Remove unnecessary includes
	sed -i.cbindgen '/#include\s*<.*[^t].h>/d' $(TARGET_PATH)/debug/libclgeom.h

target/release/libclgeom.h: $(FFI_FILES)
	cbindgen -o $(TARGET_PATH)/release/libclgeom.h
	# Remove unnecessary includes
	sed -i.cbindgen '/#include\s*<.*[^t].h>/d' $(TARGET_PATH)/release/libclgeom.h

test: build $(C_DEPS)
	$(CC) $(C_SRC_PATH)/test_main.c -o $(C_SRC_PATH)/test_libclgeom -lc -lclgeom -Ltarget/debug
	LD_LIBRARY_PATH=$(TARGET_PATH)/debug/ $(C_SRC_PATH)/test_libclgeom
