set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name InductorOpt)

add_mlir_dialect(InductorOps inductor)
add_mlir_doc(InductorDialect InductorDialect Inductor/ -gen-dialect-doc)
add_mlir_doc(InductorOps InductorOps Inductor/ -gen-op-doc)

add_public_tablegen_target(MLIRInductorPassIncGen)

add_mlir_doc(Passes InductorPasses Inductor/ -gen-pass-doc)

add_dependencies(mlir-headers MLIRInductorOpsIncGen)
add_dependencies(mlir-headers MLIRInductorPassIncGen)