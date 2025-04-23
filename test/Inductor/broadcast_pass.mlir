
// RUN: inductor-opt --broadcast-pass  %s | FileCheck %s
// CHECK-LABEL: test_sequential_ops
func.func @test_sequential_ops(%a: tensor<2xf32>, %b: tensor<3x4x2xf32>, %c: tensor<3x4x1xf32>) -> tensor<3x4x2xf32> {
  %0 = inductor.add %a, %b : (tensor<2xf32>, tensor<3x4x2xf32>) -> tensor<3x4x2xf32>
  %1 = inductor.sub %0, %c : (tensor<3x4x2xf32>, tensor<3x4x1xf32>) -> tensor<3x4x2xf32>
  %2 = inductor.entr %1 : (tensor<3x4x2xf32>) -> tensor<3x4x2xf32>
  return %2 : tensor<3x4x2xf32>
  //CHECK: %0 = inductor.reshape %arg0 {shapes = array<i64: 1, 1, 2>} : (tensor<2xf32>) -> tensor<1x1x2xf32>
  //CHECK-NEXT:  %1 = inductor.tile %0 {shapes = array<i64: 3, 4, 1>} : (tensor<1x1x2xf32>) -> tensor<3x4x2xf32>
  //CHECK-NEXT:  %2 = inductor.add %1, %arg1 : (tensor<3x4x2xf32>, tensor<3x4x2xf32>) -> tensor<3x4x2xf32>
  //CHECK-NEXT:  %3 = inductor.tile %arg2 {shapes = array<i64: 1, 1, 2>} : (tensor<3x4x1xf32>) -> tensor<3x4x2xf32>
  //CHECK-NEXT:  %4 = inductor.sub %2, %3 : (tensor<3x4x2xf32>, tensor<3x4x2xf32>) -> tensor<3x4x2xf32>
  //CHECK-NEXT:  %5 = inductor.entr %4 : (tensor<3x4x2xf32>) -> tensor<3x4x2xf32>
  //CHECK-NEXT:  return %5 : tensor<3x4x2xf32>
}


// CHECK-LABEL: test_invalid_broadcast_pass
func.func @test_invalid_broadcast_pass(%a:tensor<2x5x3xf32>) -> tensor<2x5x3xf32>{
  %0 = inductor.entr %a : (tensor<2x5x3xf32>) -> tensor<2x5x3xf32>
  return %0: tensor<2x5x3xf32>
  //CHECK: %0 = inductor.entr %arg0 : (tensor<2x5x3xf32>) -> tensor<2x5x3xf32>
  //CHECK-NEXT:   return %0 : tensor<2x5x3xf32>
}

// CHECK-LABEL: test_valid_broadcasted_form
func.func @test_valid_broadcasted_form(%a: tensor<2x2xf32>,%b:tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = inductor.add %a, %b:(tensor<2x2xf32>,tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
    //CHECK:  %0 = inductor.add %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    //CHECK-NEXT: return %0 : tensor<2x2xf32>
} 

// CHECK-LABEL: test_valid_broadcasted_pass_with_reshape_tile
func.func @test_valid_broadcasted_pass_with_reshape_tile(%a: tensor<5x1xf32>,%b:tensor<6xf32>) -> tensor<5x6xf32> {
    %0 = inductor.sub %a, %b:(tensor<5x1xf32>,tensor<6xf32>) -> tensor<5x6xf32>
    return %0 : tensor<5x6xf32>
    //CHECK: %0 = inductor.reshape %arg1 {shapes = array<i64: 1, 6>} : (tensor<6xf32>) -> tensor<1x6xf32>
    //CHECK-NEXT: %1 = inductor.tile %arg0 {shapes = array<i64: 1, 6>} : (tensor<5x1xf32>) -> tensor<5x6xf32>
    //CHECK-NEXT: %2 = inductor.tile %0 {shapes = array<i64: 5, 1>} : (tensor<1x6xf32>) -> tensor<5x6xf32>
    //CHECK-NEXT: %3 = inductor.sub %1, %2 : (tensor<5x6xf32>, tensor<5x6xf32>) -> tensor<5x6xf32>
    //CHECK-NEXT: return %3 : tensor<5x6xf32>
}

func.func @broadcasted_pass(%a: tensor<5x1xf32>,%b:tensor<6xf32>) -> tensor<5x6xf32> {
    %0 = inductor.add %a, %b:(tensor<5x1xf32>,tensor<6xf32>) -> tensor<5x6xf32>
    return %0 : tensor<5x6xf32>
}

// CHECK-LABEL: test_valid_broadcasted_pass_with_tile
func.func @test_valid_broadcasted_pass_with_tile(%a: tensor<2x1xf32>,%b:tensor<1x2xf32>) -> tensor<2x2xf32> {
    %0 = inductor.add %a, %b:(tensor<2x1xf32>,tensor<1x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
    //CHECK:  %0 = inductor.tile %arg0 {shapes = array<i64: 1, 2>} : (tensor<2x1xf32>) -> tensor<2x2xf32>
    //CHECK-NEXT: %1 = inductor.tile %arg1 {shapes = array<i64: 2, 1>} : (tensor<1x2xf32>) -> tensor<2x2xf32>
    //CHECK-NEXT: %2 = inductor.add %0, %1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    //CHECK-NEXT: return %2 : tensor<2x2xf32>
}

// CHECK-LABEL: test_valid_broadcasted_pass_with_tile_input2
func.func @test_valid_broadcasted_pass_with_tile_input2(%a: tensor<5x6xf32>,%b:tensor<1x6xf32>) -> tensor<5x6xf32> {
    %0 = inductor.add %a, %b:(tensor<5x6xf32>,tensor<1x6xf32>) -> tensor<5x6xf32>
    return %0 : tensor<5x6xf32>
    //CHECK:  %0 = inductor.tile %arg1 {shapes = array<i64: 5, 1>} : (tensor<1x6xf32>) -> tensor<5x6xf32>
    //CHECK-NEXT: %1 = inductor.add %arg0, %0 : (tensor<5x6xf32>, tensor<5x6xf32>) -> tensor<5x6xf32>
    //CHECK-NEXT: return %1 : tensor<5x6xf32>
}

// CHECK-LABEL: test_valid_broadcasted_pass_with_tile_input1
func.func @test_valid_broadcasted_pass_with_tile_input1(%a: tensor<1x6xf32>,%b:tensor<5x6xf32>) -> tensor<5x6xf32> {
    %0 = inductor.add %a, %b:(tensor<1x6xf32>,tensor<5x6xf32>) -> tensor<5x6xf32>
    return %0 : tensor<5x6xf32>
    //CHECK: %0 = inductor.tile %arg0 {shapes = array<i64: 5, 1>} : (tensor<1x6xf32>) -> tensor<5x6xf32>
    //CHECK-NEXT: %1 = inductor.add %0, %arg1 : (tensor<5x6xf32>, tensor<5x6xf32>) -> tensor<5x6xf32>
    //CHECK-NEXT: return %1 : tensor<5x6xf32>
}