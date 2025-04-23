// RUN: inductor-opt --broadcast-pass --inductor-to-tosa --tosa-to-llvm %s | FileCheck %s

// CHECK-LABEL: @test_reshape
func.func @test_reshape(%a: tensor<4x3x3xf32>) -> tensor<9x1x4xf32> {
    %0 = inductor.reshape %a { shapes = array<i64: 9,1,4>} : (tensor<4x3x3xf32>) -> tensor<9x1x4xf32>
    return %0 : tensor<9x1x4xf32>   
    //CHECK: %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %5 = llvm.insertvalue %arg6, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %7 = llvm.insertvalue %arg7, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %8 = llvm.insertvalue %arg5, %7[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %9 = llvm.insertvalue %arg8, %8[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %10 = llvm.mlir.constant(4 : index) : i64
    //CHECK: %11 = llvm.mlir.constant(3 : index) : i64
    //CHECK: %12 = llvm.mlir.constant(3 : index) : i64
    //CHECK: %20 = llvm.add %18, %19 : i64
    //CHECK: %21 = llvm.call @malloc(%20) : (i64) -> !llvm.ptr
}
