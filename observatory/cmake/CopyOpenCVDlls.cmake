# On Windows, OpenCV's prebuilt package ships its DLLs in a
# <OpenCV_LIB_PATH>/../bin directory that's nowhere on PATH by default. Unlike
# FetchOnnxRuntime.cmake's fetched tree, this is a system-wide install this
# project doesn't own, so mirror observatory_copy_onnxruntime_dlls() here
# instead of copying at fetch time: executables need OpenCV's DLLs copied
# alongside, or they fail at startup with STATUS_DLL_NOT_FOUND.
if(WIN32)
  function(observatory_copy_opencv_dlls target)
    file(GLOB _observatory_opencv_dlls "${OpenCV_LIB_PATH}/../bin/opencv_*.dll")
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_observatory_opencv_dlls} "$<TARGET_FILE_DIR:${target}>"
    )
  endfunction()
else()
  function(observatory_copy_opencv_dlls target)
  endfunction()
endif()
