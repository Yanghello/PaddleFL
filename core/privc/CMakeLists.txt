set(PRIVC_SRCS
    "privc_context.cc"
    "integer.cc"
    "bit.cc"
    "ot.cc"
    "../paddlefl_mpc/mpc_protocol/context_holder.cc"
)

add_library(privc_o OBJECT ${PRIVC_SRCS})
add_dependencies(privc_o crypto privc3)

add_library(privc SHARED $<TARGET_OBJECTS:privc_o>)

target_link_libraries(privc psi privc3)

#set(CMAKE_BUILD_TYPE Debug)

cc_test(crypto_test SRCS crypto_test.cc DEPS privc)
cc_test(privc_fixedpoint_tensor_test SRCS fixedpoint_tensor_test.cc DEPS privc)
cc_test(triplet_generator_test SRCS triplet_generator_test.cc DEPS privc)
cc_test(privc_gc_test SRCS privc_gc_test.cc DEPS privc)

