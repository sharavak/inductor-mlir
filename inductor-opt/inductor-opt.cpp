#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Inductor/InductorDialect.h"
#include "Inductor/InductorPasses.h"




 
                               


void inductorToTOSAPipelineBuilder(mlir::OpPassManager &manager){
  manager.addPass(inductor::createLowerToTosaPass());
}





int main(int argc, char **argv) {
    mlir::MLIRContext context;
    mlir::DialectRegistry registry;
    registry.insert<inductor::InductorDialect>();
    registry.insert<mlir::tosa::TosaDialect>();
    mlir::registerAllDialects(registry);
    context.getOrLoadDialect<inductor::InductorDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::registerAllPasses();
    
    return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Inductor Pass Driver", registry));
}
    