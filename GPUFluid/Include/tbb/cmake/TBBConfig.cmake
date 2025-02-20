# Copyright (c) 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# It defines the following variables:
#     TBB_<component>_FOUND
#     TBB_IMPORTED_TARGETS
#
# TBBConfigVersion.cmake defines TBB_VERSION
#
# Initialize to default values
if (NOT TBB_IMPORTED_TARGETS)
    set(TBB_IMPORTED_TARGETS "")
endif()

if (NOT TBB_FIND_COMPONENTS)
    set(TBB_FIND_COMPONENTS "tbb;tbbmalloc;tbbmalloc_proxy")
    foreach (_tbb_component ${TBB_FIND_COMPONENTS})
        set(TBB_FIND_REQUIRED_${_tbb_component} 1)
    endforeach()
endif()

# Add components with internal dependencies: tbbmalloc_proxy -> tbbmalloc
list(FIND TBB_FIND_COMPONENTS tbbmalloc_proxy _tbbmalloc_proxy_ix)
if (NOT _tbbmalloc_proxy_ix EQUAL -1)
    list(FIND TBB_FIND_COMPONENTS tbbmalloc _tbbmalloc_ix)
    if (_tbbmalloc_ix EQUAL -1)
        list(APPEND TBB_FIND_COMPONENTS tbbmalloc)
        set(TBB_FIND_REQUIRED_tbbmalloc ${TBB_FIND_REQUIRED_tbbmalloc_proxy})
    endif()
endif()

set(TBB_INTERFACE_VERSION 11102)

get_filename_component(_tbb_root "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_tbb_root "${_tbb_root}" PATH)

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(_tbb_arch_subdir intel64)
else()
    set(_tbb_arch_subdir ia32)
endif()

if (NOT MSVC)
    message(FATAL_ERROR "This Intel TBB package is intended to be used only in the project with MSVC")
endif()

if (MSVC_VERSION VERSION_LESS 1900)
    message(FATAL_ERROR "This Intel TBB package is intended to be used only in the project with MSVC version 1900 (vc14) or higher")
endif()

set(_tbb_compiler_subdir vc14)

if (WINDOWS_STORE)
    set(_tbb_compiler_subdir ${_tbb_compiler_subdir}_uwp)
endif()

get_filename_component(_tbb_lib_path "${_tbb_root}/bin/${_tbb_arch_subdir}/${_tbb_compiler_subdir}" ABSOLUTE)

foreach (_tbb_component ${TBB_FIND_COMPONENTS})
    set(TBB_${_tbb_component}_FOUND 0)

    set(_tbb_release_lib "${_tbb_lib_path}/${_tbb_component}.dll")

    if (NOT TBB_FIND_RELEASE_ONLY)
        set(_tbb_debug_lib "${_tbb_lib_path}/${_tbb_component}_debug.dll")
    endif()

    if (EXISTS "${_tbb_release_lib}" OR EXISTS "${_tbb_debug_lib}")
        if (NOT TARGET TBB::${_tbb_component})
            add_library(TBB::${_tbb_component} SHARED IMPORTED)
            set_target_properties(TBB::${_tbb_component} PROPERTIES
                                INTERFACE_INCLUDE_DIRECTORIES "${_tbb_root}/include"
                              INTERFACE_COMPILE_DEFINITIONS "__TBB_NO_IMPLICIT_LINKAGE=1")

            if (EXISTS "${_tbb_release_lib}")
                set_target_properties(TBB::${_tbb_component} PROPERTIES
                                    IMPORTED_LOCATION_RELEASE "${_tbb_release_lib}"
                                  IMPORTED_IMPLIB_RELEASE "${_tbb_root}/lib/${_tbb_arch_subdir}/${_tbb_compiler_subdir}/${_tbb_component}.lib")
                set_property(TARGET TBB::${_tbb_component} APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
            endif()

            if (EXISTS "${_tbb_debug_lib}")
                set_target_properties(TBB::${_tbb_component} PROPERTIES
                                    IMPORTED_LOCATION_DEBUG "${_tbb_debug_lib}"
                                  IMPORTED_IMPLIB_DEBUG "${_tbb_root}/lib/${_tbb_arch_subdir}/${_tbb_compiler_subdir}/${_tbb_component}_debug.lib")
                set_property(TARGET TBB::${_tbb_component} APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
            endif()

            # Add internal dependencies for imported targets: TBB::tbbmalloc_proxy -> TBB::tbbmalloc
            if (_tbb_component STREQUAL tbbmalloc_proxy)
                set_target_properties(TBB::tbbmalloc_proxy PROPERTIES INTERFACE_LINK_LIBRARIES TBB::tbbmalloc)
            endif()
        endif()
        list(APPEND TBB_IMPORTED_TARGETS TBB::${_tbb_component})
        set(TBB_${_tbb_component}_FOUND 1)
    elseif (TBB_FIND_REQUIRED AND TBB_FIND_REQUIRED_${_tbb_component})
        message(STATUS "Missed required Intel TBB component: ${_tbb_component}")
        if (TBB_FIND_RELEASE_ONLY)
            message(STATUS "  ${_tbb_release_lib} must exist.")
        else()
            message(STATUS "  one or both of:\n   ${_tbb_release_lib}\n    ${_tbb_debug_lib}\n   files must exist.")
        endif()
        set(TBB_FOUND FALSE)
    endif()
endforeach()
list(REMOVE_DUPLICATES TBB_IMPORTED_TARGETS)

unset(_tbb_arch_subdir)
unset(_tbb_compiler_subdir)
unset(_tbbmalloc_proxy_ix)
unset(_tbbmalloc_ix)
unset(_tbb_lib_path)
unset(_tbb_release_lib)
unset(_tbb_debug_lib)
