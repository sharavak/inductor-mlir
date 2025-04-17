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

using namespace mlir;
namespace inductor {
  
  #define GEN_PASS_DEF_ANALYSEBROADCASTPASS // from Pasess.h.inc
  #include "Inductor/Passes.h.inc"  
} 
  

namespace{
  
struct AnalyseBroadcastPass : inductor::impl::AnalyseBroadcastPassBase<AnalyseBroadcastPass> {
  using AnalyseBroadcastPassBase::AnalyseBroadcastPassBase;

  void runOnOperation() override{

    int totalNoOfOps=0;

    Operation *ops = getOperation();
    ops->walk([&](Operation *op) {
      if (isa<mlir::func::FuncOp>(op) || isa<mlir::func::ReturnOp>(op)) {
        return WalkResult::skip();
      }
      if (op->hasTrait<inductor::InductorBroadcastTrait>()) {
        Location loc=op->getLoc();
        totalNoOfOps++;
        
        
        llvm::outs()<<"Operation name: "<<op->getName()<<"\n";
        llvm::outs()<<"Number of operands :"<<op->getNumOperands()<<"\n";
        llvm::outs()<<"Number of results :"<<op->getNumResults()<<"\n";
        
      }else{
        llvm::outs()<<"Broadcast is not supported for this operation: "<< op->getName()<<"\n";
      }
      return WalkResult::advance();
    });
    llvm::outs()<<"Total No of broadcastable ops present :"<<totalNoOfOps<<"\n";
  }
};

}
std::unique_ptr<mlir::Pass> inductor::createInductorAnalyseBroadcastPass() {
  return std::make_unique<AnalyseBroadcastPass>();
}
