#ifndef LIB_TRANSFORM_BROADCAST_PASSES_TD_
#define LIB_TRANSFORM_BROADCAST_PASSES_TD_

include "mlir/Pass/PassBase.td"
//"mlir::func::FuncOp"
def AnalyseBroadcastPass : Pass<"inductor-analyse-broadcastable"> {
  let summary = "Analyses the op for broadcastabality trait check";
  let description = [{
   This pass is used for the analyzes purpose. It analyses whether the Op has the broadcast Trait and then it prints the op information
  }];
}


def BroadcastPass : Pass<"inductor-make-broadcastable"> {
  let summary = "applies broadcast operation for all elementwise operation";
  let description = [{
   This pass is used for transformation purposes. It ensures that all elementwise operations 
    apply broadcasting where necessary.  
  }];
}




#endif  // LIB_TRANSFORM_BROADCAST_PASSES_TD_