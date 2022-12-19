cmake_minimum_required(VERSION 3.4.1)

project(SimpleRKNN)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	message(STATUS "64bit")
	set(LIB_ARCH lib64)
else()
	message(STATUS "32bit")
	set(LIB_ARCH lib)
endif()

option(RKNN_DEVICE_BUFFER "buffer option" ON)



if (RKNN_DEVICE_BUFFER)
	add_definitions(RKNN_DEVICE_BUFFER)
endif()

set(INCLUDE_PATH ${CMAKE_SOURCE_DIR}/include)
set(MODULE_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/module/include)
set(MODULE_LIB_PATH ${CMAKE_SOURCE_DIR}/module/lib)
set(IMPL_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/lib/include)

include_directories(${RKNN_API_PATH} ${RKNN_API_PATH} ${MODULE_INCLUDE_PATH} ${IMPL_INCLUDE_PATH})

# rknn api
set(RKNN_API_PATH ${MODULE_INCLUDE_PATH}/rknn)
set(RKNN_API_LIB ${MODULE_LIB_PATH}/rknn/${CMAKE_SYSTEM_NAME}/${LIB_ARCH}/librknn_api.so)

# (${RKNN_API_LIB})

set(CMAKE_INSTALL_RPATH "lib")

add_library(SimpleRKNN
			lib/rknn_queue.cpp
			lib/simple_rknn.cpp
)

target_include_directories(SimpleRKNN PUBLIC 
        include
        ${MODULE_INCLUDE_PATH}
)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	target_link_libraries(SimpleRKNN
		${RKNN_API_LIB}
		pthread
		dl
	)
else()
	target_link_libraries(SimpleRKNN
		${RKNN_API_LIB}
		dl
	)
endif()

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/bin)

install(TARGETS SimpleRKNN DESTINATION ./)
# install(DIRECTORY model DESTINATION ./)
install(PROGRAMS ${RKNN_API_LIB} DESTINATION .)

