get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRInductor
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIROptLib
        MLIRPass
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRInductorTransforms
        MLIRTosaDialect
        InductorDialectTrait
        
        
        )
add_llvm_executable(inductor-opt inductor-opt.cpp)
# InductorDialectTrait
llvm_update_compile_flags(inductor-opt)
target_link_libraries(inductor-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(inductor-opt)