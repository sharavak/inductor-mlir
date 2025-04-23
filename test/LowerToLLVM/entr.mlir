//RUN: inductor-opt --broadcast-pass --inductor-to-tosa --tosa-to-llvm %s | FileCheck  %s

//CHECK-LABEL: @test_entr
func.func @test_entr(%a: tensor<5x2x3xf32>) -> tensor<5x2x3xf32> {
    %0 = inductor.entr %a :(tensor<5x2x3xf32>) -> tensor<5x2x3xf32>
    return %0 : tensor<5x2x3xf32>
    //CHECK: %10 = llvm.mlir.constant(3 : index) : i64
    //CHECK: %11 = llvm.mlir.constant(2 : index) : i64
    //CHECK: %12 = llvm.mlir.constant(1 : index) : i64
    //CHECK: %21 = llvm.mlir.zero : !llvm.ptr
    //CHECK: %366 = llvm.sub %361, %365 : i64
    //CHECK: %367 = llvm.add %364, %366 : i64
   
}

