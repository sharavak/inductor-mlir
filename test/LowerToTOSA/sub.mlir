//RUN: inductor-opt --inductor-to-tosa %s | FileCheck  %s

//CHECK-LABEL: @test_sub
func.func @test_sub(%a: tensor<2x2xf32>,%b:tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = inductor.sub %a, %b:(tensor<2x2xf32>,tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
    //CHECK:      %0 = tosa.sub %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    //CHECK-NEXT: return %0 : tensor<2x2xf32>
  } 