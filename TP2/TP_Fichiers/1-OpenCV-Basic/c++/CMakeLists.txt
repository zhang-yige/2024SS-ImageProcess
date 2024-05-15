cmake_minimum_required ( VERSION 3.5)
project ( ImageProc-TP2 )

# sources extension
set(SRCEXT "cpp")

file(
    GLOB_RECURSE
    source_files
    *.${SRCEXT}
)

# required packages
find_package ( OpenCV REQUIRED )

# adding the source files to the executable
add_executable(ImageProc-TP2 main.cpp)
add_executable(ImageProc-TP2_answers main_answers.cpp)


# linking other libraries
target_link_libraries(ImageProc-TP2  ${OpenCV_LIBS})
target_link_libraries(ImageProc-TP2_answers  ${OpenCV_LIBS})