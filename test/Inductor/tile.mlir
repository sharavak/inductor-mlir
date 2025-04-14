// RUN: inductor-opt %s | FileCheck %s

// CHECK-LABEL: @test_tile
func.func @test_tile(%a: tensor<4x3x3xf32>) -> tensor<4x6x3xf32> {
    %0 = inductor.tile %a { shapes = array<i64: 1,2,1>} : (tensor<4x3x3xf32>) -> tensor<4x6x3xf32>
    return %0 : tensor<4x6x3xf32>   

    //CHECK: %0 = tosa.const_shape  {value = dense<[1, 2, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
    //CHECK-NEXT: %1 = tosa.tile %arg0, %0 : (tensor<4x3x3xf32>, !tosa.shape<3>) -> tensor<4x6x3xf32>
    //CHECK-NEXT:  return %1 : tensor<4x6x3xf32>
}

// CHECK-LABEL: @test_tile_with_same_output_shape_and_input_shape
func.func @test_tile_with_same_output_shape_and_input_shape(%a: tensor<4x6x3xf32>) -> tensor<4x6x3xf32> {
    %0 = inductor.tile %a { shapes = array<i64: 1,1,1>} : (tensor<4x6x3xf32>) -> tensor<4x6x3xf32>
    return %0 : tensor<4x6x3xf32>   
    //CHECK: return %arg0 : tensor<4x6x3xf32>
}