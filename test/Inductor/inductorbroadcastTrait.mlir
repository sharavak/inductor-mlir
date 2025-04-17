// RUN: inductor-opt %s --split-input-file -verify-diagnostics

func.func @test_add_invalid_shape(%a: tensor<2x2xf32>, %b: tensor<2x3xf32>) -> tensor<2x2xf32> {
  // expected-error@+1 {{'inductor.add' op operands not broadcast-compatible at this dim 1: 2 vs 3}}
  %0 = inductor.add %a, %b : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

func.func @test_invalid_operands(%a: tensor<5x3x2xf32>, %b: tensor<5x3x2xf32>, %c: tensor<5x3x2xf32>) -> tensor<5x3x2xf32> {
  // expected-error@+1 {{'inductor.add' op expected 2 operands, but found 3}}
  %0 = inductor.add %a, %b, %c : (tensor<5x3x2xf32>, tensor<5x3x2xf32>, tensor<5x3x2xf32>) -> tensor<5x3x2xf32>
  return %0 : tensor<5x3x2xf32>
}

// -----

func.func @test_add_invalid_dtypes(%a: f32, %b: f32) -> f32 {
  // expected-error@+1 {{'inductor.add' op Only Tensor type is supported}}
  %0 = inductor.add %a, %b : (f32, f32) -> f32
  return %0 : f32
}

// -----

func.func @test_add_invalid_return_type(%a: tensor<5x3x2xf32>, %b: tensor<5x3x2xf32>) -> f32 {
  // expected-error@+1 {{'inductor.add' op Resultant type is not supported.Only tensor type is supported}}
  %0 = inductor.add %a, %b : (tensor<5x3x2xf32>, tensor<5x3x2xf32>) -> f32
  return %0 : f32
}

// -----

func.func @test_add_invalid_resultant_shape(%a: tensor<5x3x2xf32>, %b: tensor<5x3x2xf32>) ->  tensor<5x3x1xf32> {
  // expected-error@+1 {{'inductor.add' op broadcasted shape mismatches with resultant output shape}}
  %0 = inductor.add %a, %b : (tensor<5x3x2xf32>, tensor<5x3x2xf32>) -> tensor<5x3x1xf32>
  return %0 : tensor<5x3x1xf32>
}

// -----



func.func @test_add_broadcastTrait_valid_shape(%a: tensor<2x2x5xf32>, %b: tensor<2x1x5xf32>) -> tensor<2x2x5xf32> {
  %0 = inductor.add %a, %b : (tensor<2x2x5xf32>, tensor<2x1x5xf32>) -> tensor<2x2x5xf32>
  return %0 : tensor<2x2x5xf32>
}

// -----


func.func @test_add_broadcastTrait_valid_shape_with_unknown_rank(%a: tensor<2x2x5xf32>, %b: tensor<*xf32>) -> tensor<2x2x5xf32> {
  %0 = inductor.add %a, %b : (tensor<2x2x5xf32>, tensor<*xf32>) -> tensor<2x2x5xf32>
  return %0 : tensor<2x2x5xf32>
}

// -----

func.func @test_add_broadcastTrait_valid_unknown_shape(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = inductor.add %a, %b : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @test_add_broadcastTrait_unknown_rank_shape(%a: tensor<*xf32>, %b: tensor<*xf32>) -> tensor<*xf32> {
  %0 = inductor.add %a, %b : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
