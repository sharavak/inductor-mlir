
// RUN: inductor-opt --broadcast-pass --inductor-to-tosa %s | FileCheck %s
// CHECK-LABEL: test_sequential_ops
func.func @test_sequential_ops(%a: tensor<3x3xf32>,
                                 %b: tensor<1xf32>,
                                 %c: tensor<1x3xf32>,
                                 %d: tensor<3x1xf32>,
                                 %e: tensor<3xf32>,
                                 %f: tensor<1x5x2xf32>,
                                 %g: tensor<4x5x1xf32>,
                                 %h: tensor<4x1x3xf32>) -> tensor<4x5x3xf32> {
    %0 = inductor.add %a, %b : (tensor<3x3xf32>, tensor<1xf32>) -> tensor<3x3xf32>

    %1 = inductor.add %c, %d : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<3x3xf32>

    %2 = inductor.add %1, %e : (tensor<3x3xf32>, tensor<3xf32>) -> tensor<3x3xf32>

    %3 = inductor.sub %f, %g : (tensor<1x5x2xf32>, tensor<4x5x1xf32>) -> tensor<4x5x2xf32>

    %4 = inductor.sub %g, %h : (tensor<4x5x1xf32>, tensor<4x1x3xf32>) -> tensor<4x5x3xf32>

    %5 = inductor.entr %4 : (tensor<4x5x3xf32>) -> tensor<4x5x3xf32>

    %6=  inductor.add %5, %h : (tensor<4x5x3xf32>, tensor<4x1x3xf32>) -> tensor<4x5x3xf32>

    %7=  inductor.sub %6, %6 : (tensor<4x5x3xf32>,tensor<4x5x3xf32>) -> tensor<4x5x3xf32>
    
    return %7 : tensor<4x5x3xf32>

    // CHECK:  %0 = tosa.const_shape  {value = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-NEXT: %1 = tosa.reshape %arg1, %0 : (tensor<1xf32>, !tosa.shape<2>) -> tensor<1x1xf32>
    // CHECK-NEXT: %2 = tosa.const_shape  {value = dense<3> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-NEXT: %3 = tosa.tile %1, %2 : (tensor<1x1xf32>, !tosa.shape<2>) -> tensor<3x3xf32>
    // CHECK-NEXT: %4 = tosa.add %arg0, %3 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK-NEXT: %5 = tosa.const_shape  {value = dense<[3, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-NEXT: %6 = tosa.tile %arg2, %5 : (tensor<1x3xf32>, !tosa.shape<2>) -> tensor<3x3xf32>
    // CHECK-NEXT: %7 = tosa.const_shape  {value = dense<[1, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-NEXT: %8 = tosa.tile %arg3, %7 : (tensor<3x1xf32>, !tosa.shape<2>) -> tensor<3x3xf32>
    // CHECK-NEXT: %9 = tosa.add %6, %8 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK-NEXT: %10 = tosa.const_shape  {value = dense<[1, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-NEXT: %11 = tosa.reshape %arg4, %10 : (tensor<3xf32>, !tosa.shape<2>) -> tensor<1x3xf32>
    // CHECK-NEXT: %12 = tosa.const_shape  {value = dense<[3, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK-NEXT: %13 = tosa.tile %11, %12 : (tensor<1x3xf32>, !tosa.shape<2>) -> tensor<3x3xf32>
    // CHECK-NEXT: %14 = tosa.add %9, %13 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK-NEXT: %15 = tosa.const_shape  {value = dense<[4, 1, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-NEXT: %16 = tosa.tile %arg5, %15 : (tensor<1x5x2xf32>, !tosa.shape<3>) -> tensor<4x5x2xf32>
    // CHECK-NEXT: %17 = tosa.const_shape  {value = dense<[1, 1, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-NEXT: %18 = tosa.tile %arg6, %17 : (tensor<4x5x1xf32>, !tosa.shape<3>) -> tensor<4x5x2xf32>
    // CHECK-NEXT: %19 = tosa.add %16, %18 : (tensor<4x5x2xf32>, tensor<4x5x2xf32>) -> tensor<4x5x2xf32>
    // CHECK-NEXT: %20 = tosa.const_shape  {value = dense<[1, 1, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-NEXT: %21 = tosa.tile %arg6, %20 : (tensor<4x5x1xf32>, !tosa.shape<3>) -> tensor<4x5x3xf32>
    // CHECK-NEXT: %22 = tosa.const_shape  {value = dense<[1, 5, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-NEXT: %23 = tosa.tile %arg7, %22 : (tensor<4x1x3xf32>, !tosa.shape<3>) -> tensor<4x5x3xf32>
    // CHECK-NEXT: %24 = tosa.add %21, %23 : (tensor<4x5x3xf32>, tensor<4x5x3xf32>) -> tensor<4x5x3xf32>
    // CHECK-NEXT: %25 = "tosa.const"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    // CHECK-NEXT: %26 = "tosa.const"() <{value = dense<-1.000000e+00> : tensor<4x5x3xf32>}> : () -> tensor<4x5x3xf32>
    // CHECK-NEXT: %27 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<4x5x3xf32>}> : () -> tensor<4x5x3xf32>
    // CHECK-NEXT: %28 = tosa.greater %24, %27 : (tensor<4x5x3xf32>, tensor<4x5x3xf32>) -> tensor<4x5x3xi1>
    // CHECK-NEXT: %29 = tosa.equal %24, %27 : (tensor<4x5x3xf32>, tensor<4x5x3xf32>) -> tensor<4x5x3xi1>
    // CHECK-NEXT: %30 = tosa.greater %27, %24 : (tensor<4x5x3xf32>, tensor<4x5x3xf32>) -> tensor<4x5x3xi1>
    // CHECK-NEXT: %31 = tosa.select %28, %24, %27 : (tensor<4x5x3xi1>, tensor<4x5x3xf32>, tensor<4x5x3xf32>) -> tensor<4x5x3xf32>
    // CHECK-NEXT: %32 = tosa.mul %31, %26, %25 : (tensor<4x5x3xf32>, tensor<4x5x3xf32>, tensor<1xi8>) -> tensor<4x5x3xf32>
    // CHECK-NEXT: %33 = tosa.log %31 : (tensor<4x5x3xf32>) -> tensor<4x5x3xf32>
    // CHECK-NEXT: %34 = tosa.mul %32, %33, %25 : (tensor<4x5x3xf32>, tensor<4x5x3xf32>, tensor<1xi8>) -> tensor<4x5x3xf32>
    // CHECK-NEXT: %35 = tosa.select %29, %31, %27 : (tensor<4x5x3xi1>, tensor<4x5x3xf32>, tensor<4x5x3xf32>) -> tensor<4x5x3xf32>
    // CHECK-NEXT: %36 = tosa.select %30, %31, %27 : (tensor<4x5x3xi1>, tensor<4x5x3xf32>, tensor<4x5x3xf32>) -> tensor<4x5x3xf32>
    // CHECK-NEXT: %37 = tosa.add %34, %35 : (tensor<4x5x3xf32>, tensor<4x5x3xf32>) -> tensor<4x5x3xf32>
    // CHECK-NEXT: %38 = tosa.add %37, %36 : (tensor<4x5x3xf32>, tensor<4x5x3xf32>) -> tensor<4x5x3xf32>
    // CHECK-NEXT: %39 = tosa.const_shape  {value = dense<[1, 5, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK-NEXT: %40 = tosa.tile %arg7, %39 : (tensor<4x1x3xf32>, !tosa.shape<3>) -> tensor<4x5x3xf32>
    // CHECK-NEXT: %41 = tosa.add %38, %40 : (tensor<4x5x3xf32>, tensor<4x5x3xf32>) -> tensor<4x5x3xf32>
    // CHECK-NEXT: %42 = tosa.add %41, %41 : (tensor<4x5x3xf32>, tensor<4x5x3xf32>) -> tensor<4x5x3xf32>
    // CHECK-NEXT: return %42 : tensor<4x5x3xf32>
  }
