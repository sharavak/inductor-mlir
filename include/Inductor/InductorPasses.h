#ifndef MLIR_INDUCTOR_PASSES_H
#define MLIR_INDUCTOR_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"


namespace inductor {
std::unique_ptr<mlir::Pass> createLowerToTosaPass();
} 

#endif // MLIR_INDUCTOR_PASSES_H