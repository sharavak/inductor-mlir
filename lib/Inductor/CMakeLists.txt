add_mlir_dialect_library(MLIRInductor
		InductorDialect.cpp
		InductorOps.cpp
        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Inductor
        DEPENDS
        MLIRInductorOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
