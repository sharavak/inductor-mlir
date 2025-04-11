// RUN: inductor-opt %s --split-input-file 2>&1 | FileCheck %s

// CHECK: error: 'inductor.add' op operands not broadcast-compatible at this dim 1: 2 vs 3
func.func @test_fail_1(%a: tensor<2x2xf32>, %b: tensor<2x3xf32>) -> tensor<2x2xf32> {
  %0 = inductor.add %a, %b : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK: error: 'inductor.add' op operands not broadcast-compatible at this dim 0: 4 vs 2
func.func @test_fail_2(%a: tensor<4x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = inductor.add %a, %b : (tensor<4x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----


// CHECK-LABEL: test_add_broadcastTrait_valid_shape
func.func @test_add_broadcastTrait_valid_shape(%a: tensor<2x2x5xf32>,%b:tensor<2x1x5xf32>) -> tensor<2x2x5xf32> {
    %0 = inductor.add %a, %b:(tensor<2x2x5xf32>,tensor<2x1x5xf32>) -> tensor<2x2x5xf32>
    return %0 : tensor<2x2x5xf32>
}

// CHECK-LABEL: test_add_broadcastTrait_valid_unknownrank
func.func @test_add_broadcastTrait_valid_unknownrank(%a: tensor<?x?xf32>,%b:tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = inductor.add %a, %b:(tensor<?x?xf32>,tensor<?x?xf32>) -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
}


// CHECK-LABEL: test_add_broadcastTrait_unknown_rank_shape
func.func @test_add_broadcastTrait_unknown_rank_shape(%a:tensor<*xf32>,%b:tensor<*xf32>) ->tensor<*xf32>{
  %0= inductor.add %a,%b : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
  return %0: tensor<*xf32> 
}