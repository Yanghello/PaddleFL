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

// Description: implementations of each virtual op according to ABY3 protocol

#include "core/paddlefl_mpc/mpc_protocol/aby3_operators.h"
#include "core/paddlefl_mpc/operators/math/elementwise_op_function.h"

namespace paddle {
namespace mpc {

/*
template<typename T, typename DeviceContext>
using paddle::operators::math::RowwiseTransformIterator<T, DeviceContext>;

template<typename T, typename DeviceContext>
using MidWiseTransformIterator = paddle::operators::math::MidWiseTransformIterator<T, DeviceContext>;

template<typename T>
using paddle::operators::math::AddFunctor<T>;
using paddle::operators::math::GetMidDims;
*/
using namespace paddle::operators::math;
using CPUDeviceContext = paddle::platform::CPUDeviceContext;

static const size_t SHARE_NUM = 2;

void Aby3OperatorsImpl::add(const Tensor *lhs, const Tensor *rhs, Tensor *out, int axis) {
    if (lhs->dims() == rhs->dims()) {
        auto lhs_tuple = from_tensor(lhs);
        auto rhs_tuple = from_tensor(rhs);
        auto out_tuple = from_tensor(out);

        auto lhs_ = std::get<0>(lhs_tuple).get();
        auto rhs_ = std::get<0>(rhs_tuple).get();
        auto out_ = std::get<0>(out_tuple).get();

        lhs_->add(rhs_, out_);
    } else {
        Tensor in_x_t_slice;
        Tensor in_y_t_slice;
        Tensor out_t_slice;

        for (size_t i = 0; i < SHARE_NUM; ++i) {
            in_x_t_slice = lhs->Slice(i, i + 1);
            in_y_t_slice = rhs->Slice(i, i + 1);
            out_t_slice = out->Slice(i, i + 1);


            auto x_dims = in_x_t_slice.dims();
            auto y_dims = in_y_t_slice.dims();

            axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);

            PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                            "Axis should be in range [0, x_dims)");

            int pre, n, post;
            GetMidDims get_mid_dims;
            get_mid_dims(x_dims, y_dims, axis, &pre, &n, &post);

            auto x_ = in_x_t_slice.data<int64_t>();
            auto y_ = in_y_t_slice.data<int64_t>();
            auto out_ = out_t_slice.data<int64_t>();
            auto nx_ = in_x_t_slice.numel();

            paddle::platform::Transform<CPUDeviceContext> trans;
            auto cpu_device_ctx = dynamic_cast<const CPUDeviceContext*>(ContextHolder::device_ctx());
            if (post == 1) {
                trans(*cpu_device_ctx, x_, x_ + nx_,
                    RowwiseTransformIterator<int64_t, CPUDeviceContext>(y_, n),
                    out_, AddFunctor<int64_t>());
            } else {
                trans(*cpu_device_ctx, x_, x_ + nx_,
                    MidWiseTransformIterator<int64_t, CPUDeviceContext>(y_, n, post),
                    out_, AddFunctor<int64_t>());
            }
        }
    }
}

void Aby3OperatorsImpl::matmul(const Tensor *lhs, const Tensor *rhs, Tensor *out,
                               int x_num_col_dims, int y_num_col_dims) {
    auto exec_ctx = ContextHolder::exec_ctx();
    auto x_dims = lhs->dims();
    auto y_dims = rhs->dims();

    int x_mat_width = 1;
    int x_mat_height = 1;
    int y_mat_width = 1;
    int y_mat_height = 1;

    for (size_t i = 1; i < x_dims.size(); i++) {
        if (i <= x_num_col_dims) {
            x_mat_width *= x_dims[i];
        } else {
            x_mat_height *= x_dims[i];
        }
    }
    for (size_t i = 1; i < y_dims.size(); i++) {
        if (i <= y_num_col_dims) {
            y_mat_width *= y_dims[i];
        } else {
            y_mat_height *= y_dims[i];
        }
    }

    Tensor x_matrix;
    Tensor y_matrix;
    x_matrix.ShareDataWith(*lhs);
    y_matrix.ShareDataWith(*rhs);

    x_matrix.Resize({2, x_mat_width, x_mat_height});
    y_matrix.Resize({2, y_mat_width, y_mat_height});

    out->mutable_data<int64_t>(exec_ctx->GetPlace());

    auto out_dim = out->dims();
    if (out_dim.size() > 3) {
        out->Resize({2, x_mat_width, y_mat_height});
    }

    auto lhs_tuple = from_tensor(&x_matrix);
    auto rhs_tuple = from_tensor(&y_matrix);
    auto out_tuple = from_tensor(out);

    auto lhs_ = std::get<0>(lhs_tuple).get();
    auto rhs_ = std::get<0>(rhs_tuple).get();
    auto out_ = std::get<0>(out_tuple).get();

    lhs_->mat_mul(rhs_, out_);

    if (out_dim.size() > 3) {
        out->Resize(out_dim);
    }
}

} // mpc
} // paddle
