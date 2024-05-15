cmake_minimum_required ( VERSION 3.5)
project ( DFT )

# sources extension
set(SRCEXT "cpp")

file(
    GLOB_RECURSE
    source_files
    *.${SRCEXT}
)

# required packages
find_package ( OpenCV REQUIRED )

add_executable(DFT discrete_fourier_transform.cpp)

# linking other libraries
target_link_libraries(DFT  ${OpenCV_LIBS})
