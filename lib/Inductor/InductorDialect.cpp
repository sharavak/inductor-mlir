#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "Inductor/InductorDialect.h"
#include "Inductor/InductorOps.h"

using namespace mlir;
using namespace inductor;

#include "Inductor/InductorOpsDialect.cpp.inc"

void InductorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Inductor/InductorOps.cpp.inc"
      >();
}
