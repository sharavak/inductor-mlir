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

namespace cl = llvm::cl; //Command-Line Argument Handling
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input hello file>"),
                                          cl::init("-"), //Default value is -, meaning it reads from stdin if no file is provided.
                                          cl::value_desc("filename"));





int loadMLIR(mlir::MLIRContext &context,mlir::OwningOpRef<mlir::ModuleOp> &module) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
  llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }
                               
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
      return 3;
  }
  mlir::PassManager passManager(&context);
  passManager.addPass(inductor::createLowerToTosaPass());

  if (mlir::failed(passManager.run(*module))) {
    return 4;
  }
  return 0;
}


int loadAndProcessMLIR(mlir::MLIRContext &context,mlir::OwningOpRef<mlir::ModuleOp> &module) {
  if (int error = loadMLIR(context, module)){
    return error;
  }
  return 0;
}


int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "Inductor compiler\n");
    mlir::MLIRContext context;
    mlir::DialectRegistry registry;
    //registry.insert<inductor::InductorDialect>();
    registry.insert<mlir::tosa::TosaDialect>();
    mlir::registerAllDialects(registry);
    context.getOrLoadDialect<inductor::InductorDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::registerAllPasses();
    if (int error = loadAndProcessMLIR(context, module)) {
      return error;
    }
    module->print(llvm::outs());
    return 0;
}