//RUN: inductor-opt --inductor-to-llvm %s | FileCheck  %s

//CHECK-LABEL: @test_sub
func.func @test_sub(%a: tensor<2x2xf32>,%b:tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = inductor.sub %a, %b:(tensor<2x2xf32>,tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
    //CHECK: %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    //CHECK-NEXT: %1 = llvm.insertvalue %arg7, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    //CHECK-NEXT: %2 = llvm.insertvalue %arg8, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    //CHECK-NEXT: %3 = llvm.insertvalue %arg9, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    //CHECK-NEXT: %4 = llvm.insertvalue %arg10, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    //CHECK: %18 = llvm.mlir.constant(0 : index) : i64
    //CHECK-NEXT: %19 = llvm.mlir.constant(2 : index) : i64
    //CHECK-NEXT: %20 = llvm.mlir.constant(2 : index) : i64
    //CHECK-NEXT: %21 = llvm.mlir.constant(1 : index) : i64
    //CHECK-NEXT: %22 = llvm.mlir.constant(4 : index) : i64
    //CHECK: llvm.return %44 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
   
  } 