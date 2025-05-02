// RUN: inductor-opt --inductor-to-llvm %s | FileCheck %s

//CHECK-LABEL: @batchnorm_test_with_affine
func.func @batchnorm_test_with_affine(%arg0: tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32> {
    %output= inductor.batchnorm2d  %arg0 :(tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32>
    return %output : tensor<2x3x2x2xf32>
    //CHECK:%0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    //CHECK-NEXT:%1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    //CHECK-NEXT:%2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    //CHECK: %75 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    //CHECK-NEXT: %76 = llvm.mul %62, %75 : i64
    //CHECK-NEXT :%77 = llvm.add %74, %76 : i64
    //CHECK: %206 = llvm.mul %199, %14 : i64
    //CHECK-NEXT: %207 = llvm.add %206, %201 : i64
    //CHECK-NEXT: %208 = llvm.add %207, %203 : i64
    //CHECK: llvm.return %747 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
}  

//CHECK-LABEL: @batchnorm_test_without_affine
func.func @batchnorm_test_without_affine(%arg0: tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32> {
    %output= inductor.batchnorm2d  %arg0 {affine=false}:(tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32>
    return %output : tensor<2x3x2x2xf32>
    //CHECK: %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    //CHECK-NEXT: %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    //CHECK-NEXT: %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    //CHECK: %12 = llvm.mlir.constant(2 : index) : i64
    //CHECK-NEXT: %13 = llvm.mlir.constant(1 : index) : i64
    //CHECK:  %15 = llvm.mlir.constant(0 : index) : i64
    //CHECK: ^bb2:  // pred: ^bb1
    //CHECK: llvm.br ^bb3(%15 : i64)
    //CHECK: %70 = llvm.mul %58, %69 : i64
    //CHECK-NEXT: %71 = llvm.add %68, %70 : i64
    //CHECK:  %161 = llvm.getelementptr %156[%160] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    //CHECK:  %747 = llvm.insertvalue %13, %746[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    //CHECK:  llvm.return %747 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>  
}  