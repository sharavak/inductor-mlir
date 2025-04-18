#ifndef MLIR_INDUCTOR_PASSES_H
#define MLIR_INDUCTOR_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"


namespace inductor {

#define GEN_PASS_DECL
#include "Inductor/Passes.h.inc"  


std::unique_ptr<mlir::Pass> createLowerToTosaPass();
std::unique_ptr<mlir::Pass> createInductorAnalyseBroadcastPass();
std::unique_ptr<mlir::Pass> createInductorMakeBroadcastablePass();

#define GEN_PASS_REGISTRATION
#include "Inductor/Passes.h.inc"  
} 

#endif // MLIR_INDUCTOR_PASSES_H