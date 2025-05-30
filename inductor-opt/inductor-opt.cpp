#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"

#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Dialect/Linalg/Passes.h"


#include "Inductor/InductorDialect.h"
#include "Inductor/Passes.h"




void inductorToTosa(mlir::OpPassManager &manager){
  manager.addPass(inductor::createInductorMakeBroadcastablePass());
  manager.addPass(inductor::createLowerToTosaPass());
}
void inductorToLLVM(mlir::OpPassManager &manager){
  manager.addPass(inductor::createInductorMakeBroadcastablePass());
  manager.addPass(inductor::createLowerToTosaPass());

  manager.nest<mlir::func::FuncOp>().addPass(mlir::tosa::createTosaToLinalg());
  manager.nest<mlir::func::FuncOp>().addPass(mlir::tosa::createTosaToLinalgNamed());
  manager.addPass(mlir::tosa::createTosaToArith());
  manager.addPass(mlir::tosa::createTosaToTensor());

  manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));

  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                       deallocationOptions);

  manager.addPass(mlir::createConvertLinalgToLoopsPass());
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());
  manager.addPass(mlir::createLowerAffinePass());
  manager.addPass(mlir::createConvertSCFToCFPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());
  manager.addPass(mlir::createCSEPass());

}


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


    mlir::PassPipelineRegistration<> BroadcastPass(
      "broadcast-pass", "This pass will insert the necessary reshape and tile for the broadcasing",
      [](mlir::OpPassManager &pm) {
        pm.addPass(inductor::createInductorMakeBroadcastablePass());
      });  
    
    mlir::PassPipelineRegistration<>(
          "inductor-to-tosa","lowering inductor to tosa",inductorToTosa
    );
    mlir::PassPipelineRegistration<>(
          "inductor-to-llvm","lowering inductor to LLVM",inductorToLLVM);
    
    return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
    return 0;
}