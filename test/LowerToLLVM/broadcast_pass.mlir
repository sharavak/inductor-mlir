
// RUN: inductor-opt --broadcast-pass --inductor-to-tosa --tosa-to-llvm  %s | FileCheck %s
// CHECK-LABEL: test_sequential_ops
func.func @test_sequential_ops(%a: tensor<2xf32>, %b: tensor<3x4x2xf32>, %c: tensor<3x4x1xf32>) -> tensor<3x4x2xf32> {
  %0 = inductor.add %a, %b : (tensor<2xf32>, tensor<3x4x2xf32>) -> tensor<3x4x2xf32>
  %1 = inductor.sub %0, %c : (tensor<3x4x2xf32>, tensor<3x4x1xf32>) -> tensor<3x4x2xf32>
  %2 = inductor.entr %1 : (tensor<3x4x2xf32>) -> tensor<3x4x2xf32>
  return %2 : tensor<3x4x2xf32>
    //CHECK: %11 = llvm.insertvalue %arg5, %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %12 = llvm.insertvalue %arg6, %11[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %13 = llvm.insertvalue %arg7, %12[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %14 = llvm.insertvalue %arg8, %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %15 = llvm.insertvalue %arg11, %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %16 = llvm.insertvalue %arg9, %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %17 = llvm.insertvalue %arg12, %16[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %18 = llvm.insertvalue %arg10, %17[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %19 = llvm.insertvalue %arg13, %18[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %20 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK: %42 = affine.apply #map()[%41]
    //CHECK: %43 = builtin.unrealized_conversion_cast %42 : index to i64
    //CHECK: %44 = affine.apply #map()[%41]
    //CHECK: %45 = builtin.unrealized_conversion_cast %44 : index to i64
}
