cmake_minimum_required (VERSION 3.2)

set(CPP_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")


SET (BASE_SRC
        ${CPP_SOURCE_DIR}/file-to-wav.cpp
        ${CPP_SOURCE_DIR}/train-predict.cpp)



add_executable(file_to_wav ${CPP_SOURCE_DIR}/main.cpp ${BASE_SRC})

TARGET_LINK_LIBRARIES(file_to_wav sndfile)

set_property(TARGET file_to_wav PROPERTY CXX_STANDARD 11)
set_property(TARGET file_to_wav PROPERTY CXX_STANDARD_REQUIRED ON)
