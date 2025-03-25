// RUN: inductor-opt %s | FileCheck %s

//CHECK-LABEL: @test_broadcast_tensors_1
func.func @test_broadcast_tensors_1(%a: tensor<1x1x5xf32>,%b: tensor<6x5xf32>) -> (tensor<1x6x5xf32>,tensor<1x6x5xf32>) {
    %0 , %1 = inductor.broadcast_tensors %a, %b:(tensor<1x1x5xf32>,tensor<6x5xf32>) -> (tensor<1x6x5xf32>,tensor<1x6x5xf32>)
    return %0, %1 : tensor<1x6x5xf32>, tensor<1x6x5xf32>
  //CHECK:       %0 = tosa.const_shape  {value = dense<[1, 1, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
  //CHECK-NEXT:  %1 = tosa.reshape %arg0, %0 : (tensor<1x1x5xf32>, !tosa.shape<3>) -> tensor<1x1x5xf32>
  //CHECK-NEXT:  %2 = tosa.const_shape  {value = dense<[1, 6, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  //CHECK-NEXT:  %3 = tosa.tile %1, %2 : (tensor<1x1x5xf32>, !tosa.shape<3>) -> tensor<1x6x5xf32>
  //CHECK-NEXT:  %4 = tosa.const_shape  {value = dense<[1, 6, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
  //CHECK-NEXT:  %5 = tosa.reshape %arg1, %4 : (tensor<6x5xf32>, !tosa.shape<3>) -> tensor<1x6x5xf32>
  //CHECK-NEXT:  %6 = tosa.const_shape  {value = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
  //CHECK-NEXT:  %7 = tosa.tile %5, %6 : (tensor<1x6x5xf32>, !tosa.shape<3>) -> tensor<1x6x5xf32>
  //CHECK-NEXT:  return %3, %7 : tensor<1x6x5xf32>, tensor<1x6x5xf32>

} 

//CHECK-LABEL: @test_broadcast_tensors_2
func.func @test_broadcast_tensors_2(%a: tensor<3xf32>,  %b: tensor<1x4x3x1xf32>, %c: tensor<3xf32>, %d: tensor<1x4x3x1xf32>) -> (
  tensor<1x4x3x3xf32>,tensor<1x4x3x3xf32>,tensor<1x4x3x3xf32>,tensor<1x4x3x3xf32>) {
  %0, %1, %2, %3 = inductor.broadcast_tensors %a, %b, %c, %d : (tensor<3xf32>, tensor<1x4x3x1xf32>, tensor<3xf32>, tensor<1x4x3x1xf32>)-> 
    (tensor<1x4x3x3xf32>, tensor<1x4x3x3xf32>, tensor<1x4x3x3xf32>, tensor<1x4x3x3xf32>)
  return %0, %1, %2, %3 : tensor<1x4x3x3xf32>, tensor<1x4x3x3xf32>, tensor<1x4x3x3xf32>, tensor<1x4x3x3xf32>

  //CHECK:     %0 = tosa.const_shape  {value = dense<[1, 1, 1, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
  //CHECK-NEXT:  %1 = tosa.reshape %arg0, %0 : (tensor<3xf32>, !tosa.shape<4>) -> tensor<1x1x1x3xf32>
  //CHECK-NEXT:  %2 = tosa.const_shape  {value = dense<[1, 4, 3, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  //CHECK-NEXT:  %3 = tosa.tile %1, %2 : (tensor<1x1x1x3xf32>, !tosa.shape<4>) -> tensor<1x4x3x3xf32>
  //CHECK-NEXT:  %4 = tosa.const_shape  {value = dense<[1, 4, 3, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  //CHECK-NEXT:  %5 = tosa.reshape %arg1, %4 : (tensor<1x4x3x1xf32>, !tosa.shape<4>) -> tensor<1x4x3x1xf32>
  //CHECK-NEXT:  %6 = tosa.const_shape  {value = dense<[1, 1, 1, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
  //CHECK-NEXT:  %7 = tosa.tile %5, %6 : (tensor<1x4x3x1xf32>, !tosa.shape<4>) -> tensor<1x4x3x3xf32>
  //CHECK-NEXT:  %8 = tosa.const_shape  {value = dense<[1, 1, 1, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
  //CHECK-NEXT:  %9 = tosa.reshape %arg2, %8 : (tensor<3xf32>, !tosa.shape<4>) -> tensor<1x1x1x3xf32>
  //CHECK-NEXT:  %10 = tosa.const_shape  {value = dense<[1, 4, 3, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  //CHECK-NEXT:  %11 = tosa.tile %9, %10 : (tensor<1x1x1x3xf32>, !tosa.shape<4>) -> tensor<1x4x3x3xf32>
  //CHECK-NEXT:  %12 = tosa.const_shape  {value = dense<[1, 4, 3, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  //CHECK-NEXT:  %13 = tosa.reshape %arg3, %12 : (tensor<1x4x3x1xf32>, !tosa.shape<4>) -> tensor<1x4x3x1xf32>
  //CHECK-NEXT:  %14 = tosa.const_shape  {value = dense<[1, 1, 1, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
  //CHECK-NEXT:  %15 = tosa.tile %13, %14 : (tensor<1x4x3x1xf32>, !tosa.shape<4>) -> tensor<1x4x3x3xf32>
  //CHECK-NEXT:  return %3, %7, %11, %15 : tensor<1x4x3x3xf32>, tensor<1x4x3x3xf32>, tensor<1x4x3x3xf32>, tensor<1x4x3x3xf32>
}