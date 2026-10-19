# Stage the importable 'polymerfts' package in the build tree:
#   cmake -DSRC=<src/python> -DDST=<stage dir> -DCORE=<_core.so> -P stage_python.cmake
# Modules that no longer exist in SRC are removed from DST, so the staged
# package never carries files from an earlier version.
file(MAKE_DIRECTORY "${DST}")
file(GLOB _staged "${DST}/*.py")
foreach(_file ${_staged})
    get_filename_component(_name "${_file}" NAME)
    if(NOT EXISTS "${SRC}/${_name}")
        file(REMOVE "${_file}")
    endif()
endforeach()
file(GLOB _sources "${SRC}/*.py")
foreach(_file ${_sources})
    execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_file}" "${DST}")
endforeach()
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CORE}" "${DST}")
