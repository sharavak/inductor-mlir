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
    /// Recursively traverses this operation and all nested regions, blocks, and 
    /// operations. The walk can be performed in either pre-order (visit parent 
    /// before children) or post-order (visit children before parent)
    /// (default: PostOrder).
    ///
    /// The order of traversal within regions and blocks (e.g., forward or reverse) 
    /// is controlled by the Iterator template parameter, which determines how 
    /// blocks and operations at the same nesting level are visited (e.g., 
    /// ForwardIterator or ReverseIterator).
    ///
    /// Traversal Order Options
    ///   WalkOrder::PreOrder   : Visit an operation before visiting its nested ops.
    ///   WalkOrder::PostOrder  : Visit nested operations before the parent (default).
    ///   ForwardIterator       : Visit blocks/ops in lexicographic order.
    ///   ReverseIterator       : Visit blocks/ops in reverse order.
    ///
    ///   Callback Forms Supported
    ///    void(Operation*)                    : Visit every operation.
    ///    void(OpT)                           : Visit operations of a specific type.
    ///    WalkResult(Operation* | OpT)        : Visit with the ability to skip or interrupt traversal.

    ops->walk([&](Operation *op) {
      if (isa<mlir::func::FuncOp>(op) || isa<mlir::func::ReturnOp>(op)) {
        return WalkResult::skip();// the walk of the current operation, region or block and their nested elements that haven't been visited already will be skipped and will 
        //continue with the next operation, region or block.
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
      return WalkResult::advance(); //the walk will continue
    });
    llvm::outs()<<"Total No of broadcastable ops present :"<<totalNoOfOps<<"\n";
  }
};

}
std::unique_ptr<mlir::Pass> inductor::createInductorAnalyseBroadcastPass() {
  return std::make_unique<AnalyseBroadcastPass>();
}
