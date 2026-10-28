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
# A failed copy (e.g. a _core module that a running Python process keeps locked
# on Windows) fails the build, so that tests never import a stale package.
file(GLOB _sources "${SRC}/*.py")
foreach(_file ${_sources} "${CORE}")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_file}" "${DST}"
        RESULT_VARIABLE _result)
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "Could not stage ${_file} in ${DST}")
    endif()
endforeach()
