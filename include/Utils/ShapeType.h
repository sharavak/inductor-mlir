#ifndef DIALECT_INDUCTOR_UTILS
#define DIALECT_INDUCTOR_UTILS

#include "Inductor/InductorDialect.h"
#include "Inductor/InductorOps.h"
#include "Inductor/InductorPasses.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "llvm/ADT/TypeSwitch.h"

mlir::Value getConstShape(mlir::RankedTensorType tensorType,
                          mlir::SmallVector<int64_t> &targetShape,
                          mlir::PatternRewriter &rewriter,
                          mlir::Operation *op);

#endif