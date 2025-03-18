// RUN: inductor-opt %s | FileCheck %s

//CHECK-LABEL: @batchnorm_test_with_affine
func.func @batchnorm_test_with_affine(%arg0: tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32> {
    %output= inductor.batchnorm2d  %arg0 :(tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32>
    return %output : tensor<2x3x2x2xf32>
    //CHECK-NEXT:    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<2x3x2x2xf32>) -> tensor<1x3x2x2xf32>
    //CHECK-NEXT:    %1 = tosa.reduce_sum %0 {axis = 2 : i32} : (tensor<1x3x2x2xf32>) -> tensor<1x3x1x2xf32>
    //CHECK-NEXT:    %2 = tosa.reduce_sum %1 {axis = 3 : i32} : (tensor<1x3x1x2xf32>) -> tensor<1x3x1x1xf32>
    //CHECK-NEXT:    %3 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x3x1x1xf32>}> : () -> tensor<1x3x1x1xf32>
    //CHECK-NEXT:    %4 = "tosa.const"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    //CHECK-NEXT:    %5 = tosa.mul %2, %3, %4 : (tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>, tensor<1xi8>) -> tensor<1x3x1x1xf32>
    //CHECK-NEXT:    %6 = tosa.sub %arg0, %5 : (tensor<2x3x2x2xf32>, tensor<1x3x1x1xf32>) -> tensor<2x3x2x2xf32>
    //CHECK-NEXT:    %7 = tosa.reduce_sum %6 {axis = 0 : i32} : (tensor<2x3x2x2xf32>) -> tensor<1x3x2x2xf32>
    //CHECK-NEXT:    %8 = tosa.reduce_sum %7 {axis = 2 : i32} : (tensor<1x3x2x2xf32>) -> tensor<1x3x1x2xf32>
    //CHECK-NEXT:    %9 = tosa.reduce_sum %8 {axis = 3 : i32} : (tensor<1x3x1x2xf32>) -> tensor<1x3x1x1xf32>
    //CHECK-NEXT:    %10 = tosa.mul %9, %3, %4 : (tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>, tensor<1xi8>) -> tensor<2x3x2x2xf32>
    //CHECK-NEXT:    %11 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
    //CHECK-NEXT:    %12 = tosa.add %10, %11 : (tensor<2x3x2x2xf32>, tensor<1x1x1x1xf32>) -> tensor<2x3x2x2xf32>
    //CHECK-NEXT:    %13 = tosa.rsqrt %12 : (tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32>
    //CHECK-NEXT:    %14 = tosa.mul %9, %13, %4 : (tensor<1x3x1x1xf32>, tensor<2x3x2x2xf32>, tensor<1xi8>) -> tensor<2x3x2x2xf32>
    //CHECK-NEXT:    %15 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x3x1x1xf32>}> : () -> tensor<1x3x1x1xf32>
    //CHECK-NEXT:    %16 = tosa.mul %14, %15, %4 : (tensor<2x3x2x2xf32>, tensor<1x3x1x1xf32>, tensor<1xi8>) -> tensor<2x3x2x2xf32>
    //CHECK-NEXT:    %17 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x3x1x1xf32>}> : () -> tensor<1x3x1x1xf32>
    //CHECK-NEXT:    %18 = tosa.add %16, %17 : (tensor<2x3x2x2xf32>, tensor<1x3x1x1xf32>) -> tensor<2x3x2x2xf32>
    //CHECK-NEXT:    return %18 : tensor<2x3x2x2xf32>
}  

//CHECK-LABEL: @batchnorm_test_without_affine
func.func @batchnorm_test_without_affine(%arg0: tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32> {
    %output= inductor.batchnorm2d  %arg0 {affine=false}:(tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32>
    return %output : tensor<2x3x2x2xf32>
    //CHECK-NEXT:    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<2x3x2x2xf32>) -> tensor<1x3x2x2xf32>
    //CHECK-NEXT:    %1 = tosa.reduce_sum %0 {axis = 2 : i32} : (tensor<1x3x2x2xf32>) -> tensor<1x3x1x2xf32>
    //CHECK-NEXT:    %2 = tosa.reduce_sum %1 {axis = 3 : i32} : (tensor<1x3x1x2xf32>) -> tensor<1x3x1x1xf32>
    //CHECK-NEXT:    %3 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x3x1x1xf32>}> : () -> tensor<1x3x1x1xf32>
    //CHECK-NEXT:    %4 = "tosa.const"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    //CHECK-NEXT:    %5 = tosa.mul %2, %3, %4 : (tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>, tensor<1xi8>) -> tensor<1x3x1x1xf32>
    //CHECK-NEXT:    %6 = tosa.sub %arg0, %5 : (tensor<2x3x2x2xf32>, tensor<1x3x1x1xf32>) -> tensor<2x3x2x2xf32>
    //CHECK-NEXT:    %7 = tosa.reduce_sum %6 {axis = 0 : i32} : (tensor<2x3x2x2xf32>) -> tensor<1x3x2x2xf32>
    //CHECK-NEXT:    %8 = tosa.reduce_sum %7 {axis = 2 : i32} : (tensor<1x3x2x2xf32>) -> tensor<1x3x1x2xf32>
    //CHECK-NEXT:    %9 = tosa.reduce_sum %8 {axis = 3 : i32} : (tensor<1x3x1x2xf32>) -> tensor<1x3x1x1xf32>
    //CHECK-NEXT:    %10 = tosa.mul %9, %3, %4 : (tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>, tensor<1xi8>) -> tensor<2x3x2x2xf32>
    //CHECK-NEXT:    %11 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
    //CHECK-NEXT:    %12 = tosa.add %10, %11 : (tensor<2x3x2x2xf32>, tensor<1x1x1x1xf32>) -> tensor<2x3x2x2xf32>
    //CHECK-NEXT:    %13 = tosa.rsqrt %12 : (tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32>
    //CHECK-NEXT:    %14 = tosa.mul %9, %13, %4 : (tensor<1x3x1x1xf32>, tensor<2x3x2x2xf32>, tensor<1xi8>) -> tensor<2x3x2x2xf32>
    //CHECK-NEXT:    return %14 : tensor<2x3x2x2xf32>
}  