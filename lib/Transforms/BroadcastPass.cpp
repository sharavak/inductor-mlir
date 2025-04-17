#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

  mlir::Value getTileOp(ArrayRef<int64_t>finalShape,ArrayRef<int64_t>inputShape,Value input,Location loc,PatternRewriter &rewriter){
    SmallVector<int64_t>tileOutputShape;
    for (int64_t i = 0; i < finalShape.size(); i++) {
      if (inputShape[i] == 1)
        tileOutputShape.push_back(finalShape[i]);
      else
        tileOutputShape.push_back(1);
    }
    auto tileInputType = llvm::dyn_cast<RankedTensorType>(input.getType());
    auto tileOutputType = RankedTensorType::get(ArrayRef<int64_t>(finalShape), tileInputType.getElementType());
    return rewriter.create<inductor::TileOp>(loc,tileOutputType,input,tileOutputShape);
  }

  LogicalResult computeTile(PatternRewriter &rewriter, Location loc, Value &input1, Value &input2){
    auto input1Type = llvm::dyn_cast<RankedTensorType>(input1.getType());
    auto input2Type = llvm::dyn_cast<RankedTensorType>(input2.getType());

    auto input1Shape=input1Type.getShape();
    auto input2Shape=input2Type.getShape();
    int64_t input1Rank = input1Type.getRank();
    int64_t input2Rank = input2Type.getRank();

    SmallVector<int64_t>finalShape;
    for(int64_t i=0;i<input1Rank;i++)
        finalShape.push_back(std::max(input1Type.getShape()[i],input2Type.getShape()[i]));
    
    Value tpInp1=input1;
    Value tpInp2=input2;
    if(llvm::equal(finalShape,input1Shape) && llvm::equal(finalShape,input2Shape))
      return success();
    
    else if(!llvm::equal(finalShape,input1Shape) && !llvm::equal(finalShape,input2Shape)){
      tpInp1 =getTileOp(finalShape,input1Shape,tpInp1,loc,rewriter);
      tpInp2= getTileOp(finalShape,input2Shape,tpInp2,loc,rewriter);
      input1=tpInp1;
      input2=tpInp2;
      return success();
    }
    else if(llvm::equal(input1Shape,finalShape)){
      tpInp2=getTileOp(finalShape,input2Shape,tpInp2,loc,rewriter); 
      input2=tpInp2;
    }
    else if(llvm::equal(input2Shape,finalShape)){
      tpInp1=getTileOp(finalShape,input1Shape,tpInp1,loc,rewriter);
      input1=tpInp1;
    }
   
    return success();
  }


  LogicalResult computeReshape(PatternRewriter &rewriter, Location loc, Value &input1, Value &input2){
    // input1 and input2 -> from the User level
    auto input1Type = llvm::dyn_cast<RankedTensorType>(input1.getType());
    auto input2Type = llvm::dyn_cast<RankedTensorType>(input2.getType());
  
    int64_t input1Rank = input1Type.getRank();
    int64_t input2Rank = input2Type.getRank();
    
   
  
    Value highTensorVal, lowTensorVal;
    if (input1Rank > input2Rank) {
        highTensorVal = input1;
        lowTensorVal = input2;
    } else {
        highTensorVal = input2;
        lowTensorVal = input1;
      } 
    ArrayRef<int64_t> higherRankShape =
      llvm::cast<RankedTensorType>(highTensorVal.getType()).getShape();
    ArrayRef<int64_t> lowerRankShape =
      llvm::cast<RankedTensorType>(lowTensorVal.getType()).getShape();
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

  LogicalResult  getBroadcastedShape(PatternRewriter &rewriter, Location loc,RankedTensorType outputType, Value &input1,
    Value &input2,Operation *op) {
      auto input1Type = dyn_cast<RankedTensorType>(input1.getType());
      auto input2Type = dyn_cast<RankedTensorType>(input2.getType());
      int64_t input1Rank = input1Type.getRank();
      int64_t input2Rank = input2Type.getRank();
      
      Value Tempinput1 = input1;
      Value Tempinput2 = input2;

      // If the both the dim are same then tileOp is used no need of reshape
      if(input1Rank==input2Rank){

        // eg -> [1,2], [2,1] 
        if(computeTile(rewriter, loc, Tempinput1, Tempinput2).failed()){
          return failure();
        }
        input1=Tempinput1;
        input2=Tempinput2;
        return success();
      }
      if (llvm::equal(input1Type.getShape(),input2Type.getShape())) 
        return rewriter.notifyMatchFailure(loc, "Both inputs rank are same");
    
     
      if(computeReshape(rewriter, loc, Tempinput1, Tempinput2).failed()){
        return failure();
      }
      if(computeTile(rewriter, loc, Tempinput1, Tempinput2).failed()){
        return failure();
      }     
      input1=Tempinput1;
      input2=Tempinput2;
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
            auto output=op->getResult(0);
            auto outputType = llvm::dyn_cast<mlir::RankedTensorType>(output.getType());
            auto input1=op->getOperand(0);
            auto input2=op->getOperand(1);
            
            if(getBroadcastedShape(rewriter, loc, outputType,input1, input2,op).failed()){
              return WalkResult::interrupt();
            }
            rewriter.replaceOpWithNewOp<inductor::AddOp>(op,output.getType(),input1,input2);      
          }
          return WalkResult::advance();
        });
      }
    };
}
std::unique_ptr<mlir::Pass> inductor::createInductorMakeBroadcastablePass() {
    return std::make_unique<BroadcastPass>();
}