#ifndef INDUCTOR_DIALECT_BROADCASTABLETRAIT_H_
#define INDUCTOR_DIALECT_BROADCASTABLETRAIT_H_

#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"


namespace inductor {

mlir::LogicalResult verifyBroadcastability(mlir::Operation *op);

template <typename ConcreteType>
class InductorBroadcastTrait : public mlir::OpTrait::TraitBase<ConcreteType, InductorBroadcastTrait> {
 public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op){
    return inductor::verifyBroadcastability(op);
  }
};
}

#endif  // INDUCTOR_DIALECT_BROADCASTABLETRAIT_H_
