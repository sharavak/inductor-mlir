#include "Inductor/InductorTraits.h"
#include "Inductor/InductorDialect.h"
#include "Inductor/InductorOps.h"


mlir::LogicalResult inductor::verifyBroadcastability(mlir::Operation *op){

    if (op->getNumOperands() != 2)
        return op->emitError("expected two operands")<<" but found "<<op->getNumOperands();

    auto input1=op->getOperand(0);
    auto input2=op->getOperand(1);
    auto input1Type = input1.getType();
    auto input2Type = input2.getType();
    auto output = op->getResult(0);
   

    // checking either of the inputs is tensortype or not
    if(!llvm::isa<mlir::TensorType>(input1Type)|| !llvm::isa<mlir::TensorType>(input2Type))
      //Emit an error with the op name prefixed, like "'add' op " which is convenient for verifiers.
      return op->emitOpError("Only Tensor type is supported");

    if(!llvm::isa<mlir::TensorType>(output.getType()))
      return op->emitOpError("Resultant type is not supported.Only tensor type is supported");

    

  
    auto input1TensorType = llvm::dyn_cast<mlir::RankedTensorType>(input1Type);
    auto input2TensorType = llvm::dyn_cast<mlir::RankedTensorType>(input2Type);

    // checks either of the input is UnrankedTensor then return success
    if(!input1TensorType || !input2TensorType)
      return mlir::success();
    auto input1Shape = input1TensorType.getShape();
    auto input2Shape = input2TensorType.getShape();


    int64_t input1Rank = input1Shape.size();
    int64_t input2Rank = input2Shape.size();
    int64_t maxRank = std::max(input1Rank, input2Rank);

    llvm::SmallVector<int64_t> paddedinput1(maxRank, 1);
    llvm::SmallVector<int64_t> paddedinput2(maxRank, 1);

              // start range         //end range         target vector
    std::copy(input1Shape.begin(), input1Shape.end(), paddedinput1.begin() + (maxRank - input1Rank));
    std::copy(input2Shape.begin(), input2Shape.end(), paddedinput2.begin() + (maxRank - input2Rank));

    mlir::SmallVector<int64_t>resultShape;
    for (int64_t i = 0; i < maxRank; ++i) {
      int64_t l = paddedinput1[i];
      int64_t r = paddedinput2[i];
      if (l != r && l != 1 && r != 1) {
        return op->emitOpError("operands not broadcast-compatible at this dim ")
            << i << ": " << l << " vs " << r;
      }
      resultShape.push_back(std::max(paddedinput1[i],paddedinput2[i]));
    }

    auto outputShape = llvm::dyn_cast<mlir::RankedTensorType>(output.getType()).getShape();

    if(!llvm::equal(outputShape,resultShape))
      return op->emitOpError("broadcasted shape mismatches with resultant output shape");
    return mlir::success(); 
}

