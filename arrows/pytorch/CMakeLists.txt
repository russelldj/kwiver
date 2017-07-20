###
# Loadable algorithms

set(kwiver_python_subdir "python${PYTHON_VERSION}${PYTHON_ABIFLAGS}")
set(kwiver_python_output_path "${KWIVER_BINARY_DIR}/lib/${kwiver_python_subdir}")

# TODO: change KWIVER_ENABLE_PYTHON to KWIVER_ENABLE_PYTORCH
if( KWIVER_ENABLE_PYTHON )
  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/pytorch_detector.py
    kwiver/processes
    pytorch_detector)

endif()
