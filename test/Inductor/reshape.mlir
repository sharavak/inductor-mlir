// RUN: inductor-opt %s | FileCheck %s

// CHECK-LABEL: @test_reshape
func.func @test_reshape(%a: tensor<4x3x3xf32>) -> tensor<9x1x4xf32> {
    %0 = inductor.reshape %a { shapes = array<i64: 9,1,4>} : (tensor<4x3x3xf32>) -> tensor<9x1x4xf32>
    return %0 : tensor<9x1x4xf32>   
    //CHECK: %0 = tosa.const_shape  {value = dense<[9, 1, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    //CHECK-NEXT: %1 = tosa.reshape %arg0, %0 : (tensor<4x3x3xf32>, !tosa.shape<3>) -> tensor<9x1x4xf32>
    //CHECK-NEXT: return %1 : tensor<9x1x4xf32>
}

// CHECK-LABEL: @test_reshape_with_same_output_shape_and_input_shape
func.func @test_reshape_with_same_output_shape_and_input_shape(%a: tensor<9x1x4xf32>) -> tensor<9x1x4xf32> {
    %0 = inductor.reshape %a { shapes = array<i64: 9,1,4>} : (tensor<9x1x4xf32>) -> tensor<9x1x4xf32>
    return %0 : tensor<9x1x4xf32>   
    //CHECK:  return %arg0 : tensor<9x1x4xf32>
}
