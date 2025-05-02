// RUN: inductor-opt --inductor-to-llvm %s | FileCheck %s

// CHECK-LABEL: @test_reshape
func.func @test_reshape(%a: tensor<4x3x3xf32>) -> tensor<9x1x4xf32> {
    %0 = inductor.reshape %a { shapes = array<i64: 9,1,4>} : (tensor<4x3x3xf32>) -> tensor<9x1x4xf32>
    return %0 : tensor<9x1x4xf32>   
    //CHECK: %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //CHECK-NEXT: %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK-NEXT: %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK-NEXT: %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %10 = llvm.mlir.constant(4 : index) : i64
    //CHECK-NEXT: %11 = llvm.mlir.constant(3 : index) : i64
    //CHECK-NEXT: %12 = llvm.mlir.constant(1 : index) : i64
    //CHECK-NEXT: %13 = llvm.mlir.constant(9 : index) : i64
    //CHECK: %22 = llvm.sub %18, %12 : i64
    //CHECK-NEXT: %23 = llvm.add %21, %22 : i64
    //CHECK-NEXT: %24 = llvm.urem %23, %18 : i64
    //CHECK: llvm.return %59 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
}
