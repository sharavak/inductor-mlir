func.func @batchnorm_test(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32> {
    %output= inductor.batchnorm2d  %arg0: (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32>
    return %output : tensor<1x3x224x224xf32>
}  
