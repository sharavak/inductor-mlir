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
#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include "Inductor/InductorDialect.h"
#include "Inductor/Passes.h"






int main(int argc, char **argv) {
    mlir::MLIRContext context;
    mlir::DialectRegistry registry;
    registry.insert<inductor::InductorDialect>();
    registry.insert<mlir::tosa::TosaDialect>();
    mlir::registerAllDialects(registry);
    mlir::registerAllPasses();
    mlir::tosa::registerTosaOptPasses();
    context.getOrLoadDialect<inductor::InductorDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    mlir::PassPipelineRegistration<> pass1(
      "analyse-broadcast-pass", "Run analyze pass for the broadcast for the supported ops",
      [](mlir::OpPassManager &pm) {
          pm.addPass(inductor::createInductorAnalyseBroadcastPass());
      });
      mlir::PassPipelineRegistration<> pass2(
        "broadcast-pass", "This pass will insert the necessary reshape and tile for the broadcasing",
        [](mlir::OpPassManager &pm) {
          pm.addPass(inductor::createInductorMakeBroadcastablePass());
        });  
        mlir::PassPipelineRegistration<> pass3(
          "inductor-to-tosa", "lowering inductor to tosa",
          [](mlir::OpPassManager &pm) {
            pm.addPass(inductor::createLowerToTosaPass());
          });
    return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
    return 0;
}