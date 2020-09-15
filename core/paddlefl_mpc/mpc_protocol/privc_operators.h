/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Description: implementations of each op according to ABY3 protocol

#pragma once

#include <utility>

#include "context_holder.h"
#include "mpc_operators.h"
#include "paddle/fluid/framework/tensor.h"
#include "core/privc/fixedpoint_tensor.h"
#include "core/privc3/paddle_tensor.h"

namespace paddle {
namespace mpc {


using paddle::framework::Tensor;

// TODO: decide scaling factor
const size_t PRIVC_SCALING_FACTOR = PRIVC_FIXED_POINTER_SCALING_FACTOR;
using PrivCFixedTensor = privc::FixedPointTensor<int64_t, PRIVC_SCALING_FACTOR>;
using PaddleTensor = aby3::PaddleTensor<int64_t>;

using CPUDeviceContext = paddle::platform::CPUDeviceContext;

class PrivCOperatorsImpl : public MpcOperators {
public:
    void add(const Tensor *lhs, const Tensor *rhs, Tensor *out, int axis = -1) override {
        PaddleTensor lhs_(device_ctx(), *lhs);
        PaddleTensor rhs_(device_ctx(), *rhs);
        PaddleTensor out_(device_ctx(), *out);

        PrivCFixedTensor lhs_f(&lhs_);
        PrivCFixedTensor rhs_f(&rhs_);
        PrivCFixedTensor out_f(&out_);

        lhs_f.add(&rhs_f, &out_f);

    }

    // TODO: override
    void sub(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {
        PaddleTensor lhs_(device_ctx(), *lhs);
        PaddleTensor rhs_(device_ctx(), *rhs);
        PaddleTensor out_(device_ctx(), *out);

        PrivCFixedTensor lhs_f(&lhs_);
        PrivCFixedTensor rhs_f(&rhs_);
        PrivCFixedTensor out_f(&out_);

        lhs_f.sub(&rhs_f, &out_f);
    }

    void neg(const Tensor *op, Tensor *out) override {
        PaddleTensor op_(device_ctx(), *op);
        PaddleTensor out_(device_ctx(), *out);

        PrivCFixedTensor op_f(&op_);
        PrivCFixedTensor out_f(&out_);

        op_f.negative(&out_f);
    }

    void sum(const Tensor *op, Tensor *out) override {}

    void mul(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {}

    void matmul(const Tensor *lhs, const Tensor *rhs, Tensor *out,
                int x_num_col_dims = 1, int y_num_col_dims = 1) override {
        PaddleTensor lhs_(device_ctx(), *lhs);
        PaddleTensor rhs_(device_ctx(), *rhs);
        PaddleTensor out_(device_ctx(), *out);

        PrivCFixedTensor lhs_f(&lhs_);
        PrivCFixedTensor rhs_f(&rhs_);
        PrivCFixedTensor out_f(&out_);

        lhs_f.mat_mul(&rhs_f, &out_f);
    }

    void relu(const Tensor *op, Tensor *out) override {
        PaddleTensor op_(device_ctx(), *op);
        PaddleTensor out_(device_ctx(), *out);

        PrivCFixedTensor op_f(&op_);
        PrivCFixedTensor out_f(&out_);

        op_f.relu(&out_f);
    }

    void sigmoid(const Tensor *op, Tensor *out) override {
        PaddleTensor op_(device_ctx(), *op);
        PaddleTensor out_(device_ctx(), *out);

        PrivCFixedTensor op_f(&op_);
        PrivCFixedTensor out_f(&out_);

        op_f.sigmoid(&out_f);
    }

    void argmax(const Tensor *op, Tensor *out) override {
        PaddleTensor op_(device_ctx(), *op);
        PaddleTensor out_(device_ctx(), *out);

        PrivCFixedTensor op_f(&op_);
        PrivCFixedTensor out_f(&out_);

        op_f.argmax(&out_f);
    }

    void scale(const Tensor *lhs, const double factor, Tensor *out) override {}

    void relu_with_derivative(const Tensor *op, Tensor *out,
                                      Tensor *derivative) override {}

    void sigmoid_enhanced(const Tensor *op, Tensor *out) override {}

    void sigmoid_chebyshev(const Tensor *op, Tensor *out) override {}

    void softmax(const Tensor *op, Tensor *out, bool use_relu, bool use_long_div) override {}

    void gt(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {}

    void geq(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {}

    void lt(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {}

    void leq(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {}

    void eq(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {}

    void neq(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {}

    void relu_grad(const Tensor *y, const Tensor *dy, Tensor *dx, const float point) override {}

    void arith_bool_mul(const Tensor* op_a, const Tensor* op_b, Tensor* out) override {}

    void max_pooling(const Tensor* in, Tensor* out, Tensor* pos_info) override {}

    void inverse_square_root(const Tensor* in, Tensor* out) override {}

private:

    static const paddle::platform::DeviceContext* device_ctx() {
        return ContextHolder::device_ctx();
    }

};

} // mpc
} // paddle
