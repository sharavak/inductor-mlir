//RUN: inductor-opt --inductor-to-llvm %s | FileCheck  %s

func.func @test_add(%a: tensor<2x2xf32>,%b:tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = inductor.add %a, %b:(tensor<2x2xf32>,tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
    //CHECK: %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    //CHECK-NEXT: %1 = llvm.insertvalue %arg7, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    //CHECK: %15 = llvm.mlir.constant(1 : index) : i64
    //CHECK-NEXT: %16 = llvm.mlir.constant(2 : index) : i64
    //CHECK: %19 = llvm.mlir.zero : !llvm.ptr
    //CHECK: %23 = llvm.add %21, %22 : i64
    //CHECK: llvm.return %37 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

  } 