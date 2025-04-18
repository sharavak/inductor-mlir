#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Inductor/InductorDialect.h"
#include "Inductor/InductorOps.h"
#include "Inductor/Passes.h"
#include "Inductor/InductorTraits.h"


namespace inductor {
  #define GEN_PASS_DEF_BROADCASTPASS // from Pasess.h.inc
  #include "Inductor/Passes.h.inc"  
} 
  
using namespace mlir;

namespace {

  mlir::Value tileInput(ArrayRef<int64_t>tileShape, Value input, Location loc,PatternRewriter &rewriter){
    SmallVector<int64_t>multiples;
    auto inputShape=llvm::dyn_cast<mlir::RankedTensorType>(input.getType()).getShape();
    for (int64_t i = 0; i < tileShape.size(); i++) {
      if (inputShape[i] == 1)
        multiples.push_back(tileShape[i]);
      else
        multiples.push_back(1);
    }
    auto tileInputType = llvm::dyn_cast<RankedTensorType>(input.getType());
    auto tileOutputType = RankedTensorType::get(ArrayRef<int64_t>(tileShape), tileInputType.getElementType());
    return rewriter.create<inductor::TileOp>(loc,tileOutputType,input,multiples);
  }

  LogicalResult insertTileOp(PatternRewriter &rewriter, Location loc, Value &input1, Value &input2){
    auto input1Type = llvm::dyn_cast<RankedTensorType>(input1.getType());
    auto input2Type = llvm::dyn_cast<RankedTensorType>(input2.getType());

    auto input1Shape=input1Type.getShape();
    auto input2Shape=input2Type.getShape();

    SmallVector<int64_t>finalShape;
    for(int64_t i=0;i<input1Type.getRank();i++)
        finalShape.push_back(std::max(input1Type.getShape()[i],input2Type.getShape()[i]));
    if(llvm::equal(input2Shape,input1Shape))
      return success();
    
    else if(!llvm::equal(finalShape,input1Shape) && !llvm::equal(finalShape,input2Shape)){
      input1 =tileInput(finalShape,input1,loc,rewriter);
      input2= tileInput(finalShape,input2,loc,rewriter);
      return success();
    }
    else if(llvm::equal(input1Shape,finalShape)){
      input2=tileInput(finalShape,input2,loc,rewriter); 
    }
    else if(llvm::equal(input2Shape,finalShape)){
      input1=tileInput(finalShape,input1,loc,rewriter);
    }
   
    return success();
  }


  LogicalResult insertReshapeOp(PatternRewriter &rewriter, Location loc, Value &input1, Value &input2){
    // input1 and input2 -> from the User level
    auto input1Type = llvm::dyn_cast<RankedTensorType>(input1.getType());
    auto input2Type = llvm::dyn_cast<RankedTensorType>(input2.getType());
  
    int64_t input1Rank = input1Type.getRank();
    int64_t input2Rank = input2Type.getRank();
    
    Value highTensorVal, lowTensorVal;
    ArrayRef<int64_t> higherRankShape,lowerRankShape;

    if (input1Rank > input2Rank) {
        highTensorVal = input1;
        lowTensorVal = input2;
        higherRankShape = input1Type.getShape();
        lowerRankShape = input2Type.getShape();
    } else {
        highTensorVal = input2;
        lowTensorVal = input1;
        higherRankShape = input2Type.getShape();
        lowerRankShape = input1Type.getShape();
    } 
    int64_t higherRank = higherRankShape.size();
    int64_t lowerRank = lowerRankShape.size();

    SmallVector<int64_t>reshapeOutputShape;
    // for eg : [1,3] [1]
    // rankDiff=1
    int64_t rankDiff = higherRank - lowerRank;
    for(int64_t i=0;i<rankDiff;i++)
      reshapeOutputShape.push_back(1);
    for(int64_t dim:lowerRankShape)
      reshapeOutputShape.push_back(dim);

    auto reshapeInputType = llvm::dyn_cast<RankedTensorType>(lowTensorVal.getType());
    auto reshapeOutputType = RankedTensorType::get(ArrayRef<int64_t>(reshapeOutputShape), reshapeInputType.getElementType());


    auto reshapeLower = rewriter.create<inductor::ReshapeOp>(loc,reshapeOutputType, lowTensorVal, reshapeOutputShape);
    if (input1Rank > input2Rank) {
      input1 = highTensorVal;
      input2 = reshapeLower.getResult();
    } else {
      input1 = reshapeLower.getResult();
      input2 = highTensorVal;
    }

    return success();
  }

  
  LogicalResult doOpBroadcasting (PatternRewriter &rewriter, Location loc,Operation *op) {
      auto input1=op->getOperand(0);
      auto input2=op->getOperand(1);
      auto output=op->getResult(0);

      auto input1Type = dyn_cast<RankedTensorType>(input1.getType());
      auto input2Type = dyn_cast<RankedTensorType>(input2.getType());
      int64_t input1Rank = input1Type.getRank();
      int64_t input2Rank = input2Type.getRank();
      
      if(input1Rank==input2Rank){
        if(insertTileOp(rewriter, loc, input1, input2).failed()){
          return failure();
        }
        if(llvm::isa<inductor::AddOp>(op))
          rewriter.replaceOpWithNewOp<inductor::AddOp>(op,output.getType(),input1,input2); 
        else if(llvm::isa<inductor::SubOp>(op))
          rewriter.replaceOpWithNewOp<inductor::SubOp>(op,output.getType(),input1,input2);   
        return success();     
      }
      if(insertReshapeOp(rewriter, loc, input1, input2).failed()){
        return failure();
      }
      if(insertTileOp(rewriter, loc, input1, input2).failed()){
        return failure();
      }     

      if(llvm::isa<inductor::AddOp>(op))
        rewriter.replaceOpWithNewOp<inductor::AddOp>(op,output.getType(),input1,input2); 
      else if(llvm::isa<inductor::SubOp>(op))
        rewriter.replaceOpWithNewOp<inductor::SubOp>(op,output.getType(),input1,input2);
      return success();
    }
}

namespace{
  
    struct BroadcastPass : inductor::impl::BroadcastPassBase<BroadcastPass> {
      using BroadcastPassBase::BroadcastPassBase;
    
      void runOnOperation() override{    
        Operation *ops = getOperation();
        ops->walk([&](Operation *op) {
          if (isa<mlir::func::FuncOp>(op) || isa<mlir::func::ReturnOp>(op)) {
            return WalkResult::skip();
          }
          if (op->hasTrait<inductor::InductorBroadcastTrait>()) {
            Location loc=op->getLoc();
            PatternRewriter rewriter(op->getContext());
            rewriter.setInsertionPoint(op); //Sets the insertion point to the specified operation, which will cause subsequent insertions to go right before it.  
            if(doOpBroadcasting(rewriter, loc, op).failed()){
              return WalkResult::interrupt();
            }
          }
          return WalkResult::advance();
        });
      }
    };
}
std::unique_ptr<mlir::Pass> inductor::createInductorMakeBroadcastablePass() {
    return std::make_unique<BroadcastPass>();
}