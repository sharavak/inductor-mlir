//RUN: inductor-opt --inductor-to-llvm %s | FileCheck  %s

func.func @test_add(%a: tensor<2x2xf32>,%b:tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = inductor.add %a, %b:(tensor<2x2xf32>,tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
    //CHECK: %10 = llvm.insertvalue %arg1, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    //CHECK-NEXT: %11 = llvm.insertvalue %arg2, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    //CHECK-NEXT: %12 = llvm.insertvalue %arg3, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    //CHECK-NEXT: %13 = llvm.insertvalue %arg5, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    //CHECK-NEXT: %14 = llvm.insertvalue %arg4, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    //CHECK-NEXT: %15 = llvm.insertvalue %arg6, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

  } 