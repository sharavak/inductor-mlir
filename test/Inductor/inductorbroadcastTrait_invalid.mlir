// RUN: inductor-opt %s 2>&1 | FileCheck %s

// CHECK: error: 'inductor.add' op operands not broadcast-compatible at this dim 1: 2 vs 3 
func.func @test_add_broadcastTrait_invalid_shapes(%a: tensor<2x2xf32>, %b: tensor<2x3xf32>) -> tensor<2x2xf32> {
  %0 = inductor.add %a, %b : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
