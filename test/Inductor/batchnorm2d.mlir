func.func @batchnorm_test(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32> {
    %output= inductor.batchnorm2d  %arg0: (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32>
    return %output : tensor<1x3x224x224xf32>
}  

//{
      //affine=true,
      //momentum=0.1:f32,
      //track_running_stats=true,
      //eps = 0.1 : f32
    //}


//func.func @test_avg_pool2d_f32(%arg0: tensor<1x7x7x9xf32>) -> tensor<1x7x7x9xf32> {
//  %0 = tosa.avg_pool2d %arg0 {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} 
//: (tensor<1x7x7x9xf32>) -> tensor<1x7x7x9xf32>
//  return %0 : tensor<1x7x7x9xf32>
//}