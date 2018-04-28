cmake_minimum_required (VERSION 3.2)

set(CPP_SOURCE_DIR ${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")

SET (BASE_SRC
        file-to-wav.cpp)



add_executable(file_to_wav main.cpp ${BASE_SRC})

TARGET_LINK_LIBRARIES(file_to_wav sndfile)

set_property(TARGET file_to_wav PROPERTY CXX_STANDARD 11)
set_property(TARGET file_to_wav PROPERTY CXX_STANDARD_REQUIRED ON)
