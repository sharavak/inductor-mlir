add_mlir_dialect_library(MLIRInductorTransforms
		AnalyseBroadcastPass.cpp
		BroadcastPass.cpp
        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Inductor
        DEPENDS
        MLIRInductorOpsIncGen
		MLIRInductorPassIncGen
		MLIRInductor
	LINK_LIBS PUBLIC
	MLIRFuncDialect
	MLIRPass
	MLIRSupport
	MLIRIR
	)
