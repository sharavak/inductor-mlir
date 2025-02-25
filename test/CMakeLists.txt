configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(INDUCTOR_TEST_DEPENDS
        FileCheck count not
        mlir-opt
        inductor-opt
        )

add_lit_testsuite(check-inductor "Running the dl regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${INDUCTOR_TEST_DEPENDS}
        )
set_target_properties(check-inductor PROPERTIES FOLDER "Tests")

add_lit_testsuites(Inductor ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${INDUCTOR_TEST_DEPENDS})
