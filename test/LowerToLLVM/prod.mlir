// RUN: inductor-opt --inductor-to-llvm %s | FileCheck %s

// CHECK-LABEL: @test_prod_keepdim_with_dim_tuples
func.func @test_prod_keepdim_with_dim_tuples(%a: tensor<4x3x3xf32>) -> tensor<1x1x3xf32> {
    %0 = inductor.prod %a { dim = array<i64: 0,1>,keepdim=true} : (tensor<4x3x3xf32>) -> tensor<1x1x3xf32>
    return %0 : tensor<1x1x3xf32>

    //CHECK:  %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //CHECK-NEXT: %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK-NEXT: %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK-NEXT: %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %10 = llvm.mlir.constant(4 : index) : i64
    //CHECK-NEXT:%11 = llvm.mlir.constant(1 : index) : i64
    //CHECK-NEXT:%12 = llvm.mlir.constant(3 : index) : i64
    //CHECK: llvm.call @free(%74) : (!llvm.ptr) -> ()
    //CHECK-NEXT: llvm.return %143 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
}