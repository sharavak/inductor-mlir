include "InductorDialect.td"
include "InductorOps.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


def InductorBroadcastTrait : NativeOpTrait<"InductorBroadcastTrait"> {
  let cppNamespace = "::inductor";
}

