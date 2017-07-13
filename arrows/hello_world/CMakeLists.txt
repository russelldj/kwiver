###
# Loadable algorithms

set(kwiver_python_subdir "python${PYTHON_VERSION}${PYTHON_ABIFLAGS}")
set(kwiver_python_output_path "${KWIVER_BINARY_DIR}/lib/${kwiver_python_subdir}")

kwiver_create_python_plugin_init(kwiver)
kwiver_create_python_plugin_init(kwiver/processes)

if( KWIVER_ENABLE_PYTHON )
  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/hello_world_detector.py
    kwiver/processes
    hello_world_detector)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/hello_world_filter.py
    kwiver/processes
    hello_world_filter)

endif()
