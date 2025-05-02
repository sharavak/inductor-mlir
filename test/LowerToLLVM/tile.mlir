// RUN: inductor-opt --inductor-to-llvm %s | FileCheck %s

// CHECK-LABEL: @test_tile
func.func @test_tile(%a: tensor<4x3x3xf32>) -> tensor<4x6x3xf32> {
    %0 = inductor.tile %a { shapes = array<i64: 1,2,1>} : (tensor<4x3x3xf32>) -> tensor<4x6x3xf32>
    return %0 : tensor<4x6x3xf32>   

    //CHECK: %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //CHECK-NEXT: %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK-NEXT: %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK-NEXT: %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %22 = llvm.add %20, %21 : i64
    //CHECK-NEXT: %23 = llvm.call @malloc(%22) : (i64) -> !llvm.ptr
    //CHECK-NEXT: %24 = llvm.ptrtoint %23 : !llvm.ptr to i64
    //CHECK-NEXT: %25 = llvm.sub %21, %13 : i64
    //CHECK-NEXT: %26 = llvm.add %24, %25 : i64
    //CHECK: %100 = llvm.insertvalue %13, %99[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK-NEXT: llvm.return %100 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
}

