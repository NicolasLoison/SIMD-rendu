cmake_minimum_required(VERSION 3.22)
project(TP1_files C)

add_compile_options(-Wall -msse4.1 -O2)
set(CMAKE_C_STANDARD 11)

add_executable(matmul
        src/matmul.c)
add_executable(matmul_scalaire
        src/matmul_scalaire.c)
add_executable(matmul-sse
        src/matmul-sse.c)
add_executable(matmul-sse8
        src/matmul-sse8.c)
add_executable(matmul-sse16
        src/matmul-sse16.c)

enable_testing()
add_executable(tests_matmul
        tests/tests_matmul.c src/lib_matmul.h)
add_test(NAME tests_matmul COMMAND tests_matmul)
add_executable(tests_matmul_scalaire
        tests/tests_matmul_scalaire.c src/lib_matmul.h)
add_test(NAME tests_matmul_scalaire COMMAND tests_matmul_scalaire)
add_executable(tests_matmul_sse
        tests/tests_matmul_sse.c src/lib_matmul.h)
add_test(NAME tests_matmul_sse COMMAND tests_matmul_sse)
add_executable(tests_matmul_sse8
        tests/tests_matmul_sse8.c src/lib_matmul.h)
add_test(NAME tests_matmul_sse8 COMMAND tests_matmul_sse8)
add_executable(tests_matmul_sse16
        tests/tests_matmul_sse16.c src/lib_matmul.h)
add_test(NAME tests_matmul_sse16 COMMAND tests_matmul_sse16)