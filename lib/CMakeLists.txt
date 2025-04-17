add_subdirectory(Inductor)
add_subdirectory(Utils)

add_mlir_library(InductorDialectTrait
    InductorBroadcastTrait.cpp
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Inductor
    DEPENDS
    MLIRInductorOpsIncGen
	MLIRInductor
    LINK_LIBS PUBLIC
    MLIRIR
)
