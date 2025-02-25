#ifndef INDUCTOR_OPS
#define INDUCTOR_OPS


include "InductorDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"



def Inductor_AddOp : Inductor_Op<"add",[SameOperandsAndResultShape]> {
    let summary = "add";
    let description = [{
    Adds two operands, which can be either scalars or tensors of the same type.
  }];
  
    let arguments = (ins AnyTypeOf<[AnyTensor, AnyFloat, AnyInteger]>:$a, 
                       AnyTypeOf<[AnyTensor, AnyFloat, AnyInteger]>:$b);
  
    let results = (outs AnyTypeOf<[AnyTensor, AnyFloat, AnyInteger]>:$result);

    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

}

def Inductor_BatchNorm2D :Inductor_Op<"batchnorm2d",[SameOperandsAndResultShape]>{
  let summary="Batch Normalization2d";
  let description = [{
    Performs batch normalization over a 4D input tensor (N, C, H, W).
    Normalization is applied across (N, H, W) for each channel.
  }];

  let arguments=(ins AnyTensor:$input, // accepts any data type of Tensor 
                     DefaultValuedOptionalAttr<BoolAttr,"true">:$affine, // default to true
                     DefaultValuedOptionalAttr<F32Attr,"0.1" >:$momentum, // https://discourse.llvm.org/t/i-have-some-problems-designing-mlir-op/68889/3
                     DefaultValuedOptionalAttr<BoolAttr,"true">:$track_running_stats, // default to true
                     DefaultValuedOptionalAttr<F32Attr,"1e-05">:$eps
                );

  let results=(outs AnyTensor:$output); // shape should be as input shape

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

}

#endif // INDUCTOR_OPS
