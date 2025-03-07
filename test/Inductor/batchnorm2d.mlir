// RUN: inductor-opt %s | FileCheck %s

// CHECK-LABEL: func.func @batchnorm_test() -> (i32, i32)
func.func @batchnorm_test(%arg0: tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32> {
    //CHECK-NEXT:        %0 = tosa.add %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    //CHECK-NEXT:        return %0 : tensor<2x2xf32>    
    %output= inductor.batchnorm2d  %arg0:(tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32>
    return %output : tensor<2x3x2x2xf32>


}  
