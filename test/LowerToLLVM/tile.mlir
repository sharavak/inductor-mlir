// RUN: inductor-opt %s | FileCheck %s

// CHECK-LABEL: @test_tile
func.func @test_tile(%a: tensor<4x3x3xf32>) -> tensor<4x6x3xf32> {
    %0 = inductor.tile %a { shapes = array<i64: 1,2,1>} : (tensor<4x3x3xf32>) -> tensor<4x6x3xf32>
    return %0 : tensor<4x6x3xf32>   

   //CHECK: %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
   //CHECK: %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
   //CHECK: %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
   //CHECK: %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
   //CHECK: %5 = llvm.insertvalue %arg6, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
   //CHECK: %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
   //CHECK: %44 = llvm.insertvalue %43, %42[2] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)> 
   //CHECK: %45 = llvm.insertvalue %15, %44[3, 0] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)> 
   //CHECK: %46 = llvm.insertvalue %16, %45[3, 1] : !llvm.struct<(ptr, ptr, i64, array<6 x i64>, array<6 x i64>)> 
   //CHECK: llvm.return %123 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>

}

