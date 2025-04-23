// RUN: inductor-opt --broadcast-pass --inductor-to-tosa --tosa-to-llvm %s | FileCheck %s

//CHECK-LABEL: @batchnorm_test_with_affine
func.func @batchnorm_test_with_affine(%arg0: tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32> {
    %output= inductor.batchnorm2d  %arg0 :(tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32>
    return %output : tensor<2x3x2x2xf32>

    //CHECK: %53 = llvm.extractvalue %46[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %54 = llvm.mlir.constant(4 : index) : i64
    //CHECK: %55 = llvm.mul %47, %54 : i64
    //CHECK: %56 = llvm.mlir.constant(2 : index) : i64
    //CHECK: %57 = llvm.mul %49, %56 : i64
    //CHECK: %58 = llvm.add %55, %57 : i64
    //CHECK: %59 = llvm.add %58, %51 : i64

}  

//CHECK-LABEL: @batchnorm_test_without_affine
func.func @batchnorm_test_without_affine(%arg0: tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32> {
    %output= inductor.batchnorm2d  %arg0 {affine=false}:(tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32>
    return %output : tensor<2x3x2x2xf32>
    //CHECK: %100 = llvm.mul %66, %99 : i64
    //CHECK: %101 = llvm.mlir.constant(2 : index) : i64
    //CHECK: %102 = llvm.mul %68, %101 : i64
    //CHECK: %103 = llvm.add %100, %102 : i64
    //CHECK: %104 = llvm.add %103, %70 : i64
    
}  