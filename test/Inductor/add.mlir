//RUN: inductor-opt --inductor-to-tosa %s | FileCheck  %s

//CHECK-LABEL: @test_add
func.func @test_add(%a: tensor<2x2xf32>,%b:tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = inductor.add %a, %b:(tensor<2x2xf32>,tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
    //CHECK-NEXT:     %0 = tosa.add %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    //CHECK-NEXT:     return %0 : tensor<2x2xf32>

  } 