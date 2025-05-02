// RUN: inductor-opt --inductor-to-tosa %s | FileCheck %s

// CHECK-LABEL: @test_prod_keepdim_with_dim_tuples
func.func @test_prod_keepdim_with_dim_tuples(%a: tensor<4x3x3xf32>) -> tensor<1x1x3xf32> {
    %0 = inductor.prod %a { dim = array<i64: 0,1>,keepdim=true} : (tensor<4x3x3xf32>) -> tensor<1x1x3xf32>
    return %0 : tensor<1x1x3xf32>
    //CHECK: %0 = tosa.reduce_prod %arg0 {axis = 0 : i32} : (tensor<4x3x3xf32>) -> tensor<1x3x3xf32>
    //CHECK-NEXT: %1 = tosa.reduce_prod %0 {axis = 1 : i32} : (tensor<1x3x3xf32>) -> tensor<1x1x3xf32>
    //CHECK-NEXT: return %1 : tensor<1x1x3xf32>
}

// CHECK-LABEL: @test_prod_keepdim
func.func @test_prod_keepdim(%a: tensor<4x3x3xf32>) -> tensor<4x1x3xf32> {
    %0 = inductor.prod %a {dim = 1,keepdim=true} : (tensor<4x3x3xf32>) -> tensor<4x1x3xf32>
    return %0 : tensor<4x1x3xf32>

    //CHECK: %0 = tosa.reduce_prod %arg0 {axis = 1 : i32} : (tensor<4x3x3xf32>) -> tensor<4x1x3xf32>
    //CHECK-NEXT: return %0 : tensor<4x1x3xf32>
}

// CHECK-LABEL: @test_prod_not_keepdim_with_dim_tuples
func.func @test_prod_not_keepdim_with_dim_tuples(%a: tensor<4x3x3xf32>) -> tensor<3xf32> {
    %0 = inductor.prod %a {dim = array<i64: 0,1>,keepdim = false} : (tensor<4x3x3xf32>) -> tensor<3xf32>
    return %0 : tensor<3xf32>
    //CHECK: %0 = tosa.reduce_prod %arg0 {axis = 0 : i32} : (tensor<4x3x3xf32>) -> tensor<1x3x3xf32>
    //CHECK-NEXT: %1 = tosa.reduce_prod %0 {axis = 1 : i32} : (tensor<1x3x3xf32>) -> tensor<1x1x3xf32>
    //CHECK-NEXT: %2 = tosa.const_shape  {value = dense<3> : tensor<1xindex>} : () -> !tosa.shape<1>
    //CHECK-NEXT: %3 = tosa.reshape %1, %2 : (tensor<1x1x3xf32>, !tosa.shape<1>) -> tensor<3xf32>
    //CHECK-NEXT: return %3 : tensor<3xf32>
}

// CHECK-LABEL: @test_prod_not_keepdim
func.func @test_prod_not_keepdim(%a: tensor<4x3x3xf32>) -> tensor<4x3xf32> {
    %0 = inductor.prod %a {dim = 1} : (tensor<4x3x3xf32>) -> tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
    //CHECK: %0 = tosa.reduce_prod %arg0 {axis = 1 : i32} : (tensor<4x3x3xf32>) -> tensor<4x1x3xf32>
    //CHECK-NEXT: %1 = tosa.const_shape  {value = dense<[4, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
    //CHECK-NEXT: %2 = tosa.reshape %0, %1 : (tensor<4x1x3xf32>, !tosa.shape<2>) -> tensor<4x3xf32>
    //CHECK-NEXT: return %2 : tensor<4x3xf32>
}