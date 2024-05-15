cmake_minimum_required ( VERSION 3.5)
project ( MORPH )

# sources extension
set(SRCEXT "cpp")

file(
    GLOB_RECURSE
    source_files
    *.${SRCEXT}
)

# required packages
find_package ( OpenCV REQUIRED )

add_executable(morphology Morphology.cpp)

# linking other libraries
target_link_libraries(morphology ${OpenCV_LIBS})
