//RUN: inductor-opt --inductor-to-llvm %s | FileCheck  %s

//CHECK-LABEL: @test_sub
func.func @test_sub(%a: tensor<2x2xf32>,%b:tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = inductor.sub %a, %b:(tensor<2x2xf32>,tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>

    //CHECK: %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    //CHECK-NEXT: %1 = llvm.insertvalue %arg7, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    //CHECK-NEXT: %2 = llvm.insertvalue %arg8, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    //CHECK-NEXT: %3 = llvm.insertvalue %arg9, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    //CHECK: %26 = llvm.sub %22, %15 : i64
    //CHECK-NEXT: %27 = llvm.add %25, %26 : i64
    //CHECK-NEXT: %28 = llvm.urem %27, %22 : i64
    //CHECK: llvm.return %37 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  } 