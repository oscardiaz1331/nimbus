function(observatory_set_warnings target)
  if(MSVC)
    # cl.exe doesn't understand GCC/Clang -W flags (fails with D8021), so this
    # is the closest MSVC /w4xxxx equivalent per warning below:
    #   Wall/Wextra/Wpedantic -> /W4, /permissive-
    #   Wshadow               -> /w14456 /w14457 /w14458 /w14459
    #   Wnon-virtual-dtor     -> /w14265
    #   Woverloaded-virtual   -> /w14263
    #   Wconversion           -> /w14242 /w14254
    #   Wsign-conversion      -> /w14287 /w14826
    # (Wsuggest-override, Wold-style-cast have no MSVC equivalent warning.)
    target_compile_options(${target} PRIVATE
      /W4
      /permissive-
      /w14456
      /w14457
      /w14458
      /w14459
      /w14265
      /w14263
      /w14242
      /w14254
      /w14287
      /w14826
    )
    if(OBSERVATORY_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wshadow
      -Wnon-virtual-dtor
      -Woverloaded-virtual
      -Wsuggest-override
      -Wconversion
      -Wsign-conversion
      -Wold-style-cast
    )
    if(OBSERVATORY_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()
