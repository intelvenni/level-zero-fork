set(TARGET_NAME test7)

add_executable(${TARGET_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/test7_cmdlist_execute_multiple_approach.cpp
)
    
if(MSVC)
    set_target_properties(${TARGET_NAME}
        PROPERTIES
            VS_DEBUGGER_COMMAND_ARGUMENTS ""
            VS_DEBUGGER_WORKING_DIRECTORY "$(OutDir)"
    )
endif()

target_link_libraries(${TARGET_NAME}
    ${TARGET_LOADER_NAME}
    ${CMAKE_DL_LIBS}
)
