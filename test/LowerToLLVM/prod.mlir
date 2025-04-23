// RUN: inductor-opt --broadcast-pass --inductor-to-tosa --tosa-to-llvm %s | FileCheck %s

// CHECK-LABEL: @test_prod_keepdim_with_dim_tuples
func.func @test_prod_keepdim_with_dim_tuples(%a: tensor<4x3x3xf32>) -> tensor<1x1x3xf32> {
    %0 = inductor.prod %a { dim = array<i64: 0,1>,keepdim=true} : (tensor<4x3x3xf32>) -> tensor<1x1x3xf32>
    return %0 : tensor<1x1x3xf32>
    //CHECK: %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %22 = llvm.mlir.constant(64 : index) : i64
    //CHECK: %23 = llvm.add %21, %22 : i64
    //CHECK: %62 = llvm.mul %52, %61 : i64
}