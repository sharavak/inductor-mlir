// RUN: inductor-opt %s | FileCheck %s

// CHECK-LABEL: test_add_broadcastTrait_valid_shape
func.func @test_add_broadcastTrait_valid_shape(%a: tensor<2x2x5xf32>,%b:tensor<2x1x5xf32>) -> tensor<2x2x5xf32> {
    %0 = inductor.add %a, %b:(tensor<2x2x5xf32>,tensor<2x1x5xf32>) -> tensor<2x2x5xf32>
    return %0 : tensor<2x2x5xf32>
}
