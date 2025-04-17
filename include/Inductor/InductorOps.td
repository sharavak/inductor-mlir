#ifndef INDUCTOR_OPS
#define INDUCTOR_OPS


include "InductorDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "InductorTraits.td"


def Inductor_AddOp : Inductor_Op<"add",[InductorBroadcastTrait]> {
    let summary = "add";
    let description = [{
    Elementwise addition of input1 and input2 of the tensor type
  }];
  
    let arguments = (ins AnyTensor :$lhs, 
                       AnyTensor :$rhs);
  
    let results = (outs AnyTensor:$result);

    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

}

def Inductor_SubOp : Inductor_Op<"sub",[InductorBroadcastTrait]> {
    let summary = "It takes two input operands as tensor types of the same rank and subtract the two inputs";
    let description = [{
    Elementwise addition of input1 and input2 of the tensor type
  }];
  
    let arguments = (ins AnyTensor :$lhs, 
                       AnyTensor :$rhs);
  
    let results = (outs AnyTensor:$result);

    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

}


def Inductor_BatchNorm2dOp :Inductor_Op<"batchnorm2d">{
  let summary="Batch Normalization2d";
  let description = [{
    Performs batch normalization over a 4D input tensor (N, C, H, W).
    Normalization is applied across (N, H, W) for each channel.
  }];

  let arguments=(ins AnyTensor:$input, // accepts any data type of Tensor 
                     DefaultValuedOptionalAttr<BoolAttr,"true">:$affine, // default to true
                     DefaultValuedOptionalAttr<F32Attr,"0.1" >:$momentum, 
                     DefaultValuedOptionalAttr<BoolAttr,"true">:$track_running_stats, // default to true
                     OptionalAttr<F32Attr>:$eps
                );

  let results=(outs AnyTensor:$output); // shape should be as input shape

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

}
def Inductor_ProdOp: Inductor_Op<"prod">{
  let summary="prod";
  let description = [{
    Returns the product of each row of the input tensor in the given dimension dim.
  }];
  let arguments=(ins AnyTensor:$input,
                      OptionalAttr<AnyAttrOf<[ AnyI64Attr,DenseI64ArrayAttr]>>:$dim,
                      DefaultValuedOptionalAttr<BoolAttr,"false">:$keepdim
    );
  let results=(outs AnyTensor:$output);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Inductor_BroadcastTensorsOp: Inductor_Op<"broadcast_tensors">{
  let summary="broadcast_tensorsOp";
  let description = [{
    "It takes multiple inputs and broadcasts them according to broadcasting rules, making the tensors compatible for element-wise operations."
  }];
    let arguments=(ins Variadic<AnyTensor>:$inputs
    );
  let results=(outs Variadic<AnyTensor>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}                    
def Inductor_EntrOp: Inductor_Op<"entr">{
  let summary="entr";
  let description = [{
    Computes the entropy on input elementwise.
  }];
  let arguments=(ins AnyTensor:$input);
  let results=(outs AnyTensor:$output);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Inductor_TileOp: Inductor_Op<"tile">{
  let summary="tile";
  let description=[{ Computes the tile on the input}];
  let arguments=(ins AnyTensor:$input, 
                    DenseI64ArrayAttr:$shapes);
  let results =(outs AnyTensor: $output);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

}


def Inductor_ReshapeOp: Inductor_Op<"reshape">{
  let summary="reshape";
  let description=[{ Computes the reshape on the input}];
  let arguments=(ins AnyTensor:$input, 
                    DenseI64ArrayAttr:$shapes);
  let results =(outs AnyTensor: $output);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

}

#endif // INDUCTOR_OPS
