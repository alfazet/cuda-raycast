# CUDA Raycast

## Compilation and usage

Requirements:

- CMake (at least v3.10)
- GNU Make
- NVCC compiler with CUDA 12.9

```shell
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Run the binary with:

```shell
./cuda-raycast <obj_file_path> <lights_file_path> <width> <height> [cpu]
```

- `obj_file_path`: path to the .obj file containing the object to render (only a limited\
  subset of .obj syntax is supported, see comments in `obj_parser.cuh` for details)
- `lights_file_path`: path to the file specifying the position of light sources\
  and properties of the material (see comments in `light_parser.cuh` for syntax reference)
- `width`, `height`: width and height of the window (and the resolution of the rendered image)
- `cpu`: **optional** argument to launch the CPU-rendered version (very slow on larger resolutions and with larger
  objects!)

## Keys

- W/S/A/D/Q/E - rotate the object
- Up/Down/Left/Right - pan the camera
- Minus/Plus - zoom the camera
- H/L - rotate the light sources