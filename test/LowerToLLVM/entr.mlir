//RUN: inductor-opt --inductor-to-llvm %s | FileCheck  %s

//CHECK-LABEL: @test_entr
func.func @test_entr(%a: tensor<5x2x3xf32>) -> tensor<5x2x3xf32> {
    %0 = inductor.entr %a :(tensor<5x2x3xf32>) -> tensor<5x2x3xf32>
    return %0 : tensor<5x2x3xf32>
    //CHECK: %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //CHECK-NEXT: %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK-NEXT: %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK-NEXT: %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    //CHECK: %19 = llvm.mlir.addressof @__constant_5x2x3xf32 : !llvm.ptr
    //CHECK-NEXT: %20 = llvm.getelementptr %19[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x array<2 x array<3 x f32>>>
    //CHECK: %48 = llvm.sub %44, %12 : i64
    //CHECK-NEXT: %49 = llvm.add %47, %48 : i64
    //CHECK: llvm.call @free(%511) : (!llvm.ptr) -> ()
    //CHECK-NEXT: llvm.return %479 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
}

