
// RUN: inductor-opt --inductor-to-llvm  %s | FileCheck %s
// CHECK-LABEL: test_sequential_ops
func.func @test_sequential_ops(%a: tensor<2xf32>, %b: tensor<3x4x2xf32>, %c: tensor<3x4x1xf32>) -> tensor<3x4x2xf32> {
  %0 = inductor.add %a, %b : (tensor<2xf32>, tensor<3x4x2xf32>) -> tensor<3x4x2xf32>
  %1 = inductor.sub %0, %c : (tensor<3x4x2xf32>, tensor<3x4x1xf32>) -> tensor<3x4x2xf32>
  %2 = inductor.entr %1 : (tensor<3x4x2xf32>) -> tensor<3x4x2xf32>
  return %2 : tensor<3x4x2xf32>
    //CHECK: %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //CHECK-NEXT: %1 = llvm.insertvalue %arg14, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK-NEXT: %2 = llvm.insertvalue %arg15, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK-NEXT: %3 = llvm.insertvalue %arg16, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //CHECK: %25 = llvm.mlir.constant(2 : index) : i64
    //CHECK-NEXT: %26 = llvm.mlir.constant(4 : index) : i64
    //CHECK-NEXT: %27 = llvm.mlir.constant(1 : index) : i64
    //CHECK: %579 = llvm.add %578, %59 : i64
    //CHECK-NEXT: %580 = llvm.urem %579, %55 : i64
    //CHECK-NEXT: %581 = llvm.sub %579, %580 : i64
    //CHECK: llvm.return %708 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>   
}
