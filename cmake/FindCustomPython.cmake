# cmake/FindCustomPython.cmake
function(find_custom_python REQUIRED_PKG OUT_VAR)
    # 候选 Python 路径列表
    set(PYTHON_CANDIDATES
        "python3" "python"
        "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python"
        "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe"
        "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/bin/python"
        "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/Scripts/python.exe"
    )

    set(VALID_PY "")
    foreach(PY_EX_PATH ${PYTHON_CANDIDATES})
        message(STATUS "Checking Python executable: ${PY_EX_PATH}")
        find_program(TMP_PY NAMES ${PY_EX_PATH} NO_DEFAULT_PATH)
        if(NOT TMP_PY)
            find_program(TMP_PY NAMES ${PY_EX_PATH})
        endif()

        if(TMP_PY)
            # 校验所需模块
            execute_process(
                COMMAND "${TMP_PY}" -c "import ${REQUIRED_PKG}"
                RESULT_VARIABLE RET_CODE
                OUTPUT_QUIET ERROR_QUIET
            )
            if(RET_CODE EQUAL 0)
                set(VALID_PY "${TMP_PY}")
                message(STATUS "Found valid Python executable: ${VALID_PY} with required package: ${REQUIRED_PKG}")
                break()
            endif()
        endif()
        unset(TMP_PY CACHE)
    endforeach()

    if(NOT VALID_PY)
        message(FATAL_ERROR "No valid Python executable found with required package: ${REQUIRED_PKG}. Please ensure that Python is installed and the package is available.")
    endif()

    set(${OUT_VAR} "${VALID_PY}" PARENT_SCOPE)
endfunction()