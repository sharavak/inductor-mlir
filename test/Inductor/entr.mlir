//RUN: inductor-opt %s | FileCheck  %s

//CHECK-LABEL: @test_entr
func.func @test_entr(%a: tensor<5x2x3xf32>) -> tensor<5x2x3xf32> {
    %0 = inductor.entr %a :(tensor<5x2x3xf32>) -> tensor<5x2x3xf32>
    return %0 : tensor<5x2x3xf32>
    //CHECK: %0 = "tosa.const"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    //CHECK-NEXT: %1 = "tosa.const"() <{value = dense<-1.000000e+00> : tensor<5x2x3xf32>}> : () -> tensor<5x2x3xf32>
    //CHECK-NEXT: %2 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<5x2x3xf32>}> : () -> tensor<5x2x3xf32>
    //CHECK-NEXT: %3 = tosa.greater %arg0, %2 : (tensor<5x2x3xf32>, tensor<5x2x3xf32>) -> tensor<5x2x3xi1>
    //CHECK-NEXT: %4 = tosa.equal %arg0, %2 : (tensor<5x2x3xf32>, tensor<5x2x3xf32>) -> tensor<5x2x3xi1>
    //CHECK-NEXT: %5 = tosa.greater %2, %arg0 : (tensor<5x2x3xf32>, tensor<5x2x3xf32>) -> tensor<5x2x3xi1>
    //CHECK-NEXT: %6 = tosa.select %3, %arg0, %2 : (tensor<5x2x3xi1>, tensor<5x2x3xf32>, tensor<5x2x3xf32>) -> tensor<5x2x3xf32>
    //CHECK-NEXT: %7 = tosa.mul %6, %1, %0 : (tensor<5x2x3xf32>, tensor<5x2x3xf32>, tensor<1xi8>) -> tensor<5x2x3xf32>
    //CHECK-NEXT: %8 = tosa.log %6 : (tensor<5x2x3xf32>) -> tensor<5x2x3xf32>
    //CHECK-NEXT: %9 = tosa.mul %7, %8, %0 : (tensor<5x2x3xf32>, tensor<5x2x3xf32>, tensor<1xi8>) -> tensor<5x2x3xf32>
    //CHECK-NEXT: %10 = tosa.select %4, %6, %2 : (tensor<5x2x3xi1>, tensor<5x2x3xf32>, tensor<5x2x3xf32>) -> tensor<5x2x3xf32>
    //CHECK-NEXT: %11 = tosa.select %5, %6, %2 : (tensor<5x2x3xi1>, tensor<5x2x3xf32>, tensor<5x2x3xf32>) -> tensor<5x2x3xf32>
    //CHECK-NEXT: %12 = tosa.add %9, %10 : (tensor<5x2x3xf32>, tensor<5x2x3xf32>) -> tensor<5x2x3xf32>
    //CHECK-NEXT: %13 = tosa.add %12, %11 : (tensor<5x2x3xf32>, tensor<5x2x3xf32>) -> tensor<5x2x3xf32>
    //CHECK-NEXT: return %13 : tensor<5x2x3xf32>
}

