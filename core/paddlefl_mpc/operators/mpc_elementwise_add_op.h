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

// This op is different with elementwise_add of PaddlePaddle.
// We only consider that the dimensions of X is equal with the dimensions of Y.

#pragma once
#include "mpc_op.h"
#include "paddle/fluid/platform/transform.h"
#include "core/paddlefl_mpc/operators/math/elementwise_op_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename Functor, typename T, typename DeviceContext,
          typename OutType = T>
class TransformFunctor {
 public:
  TransformFunctor(const framework::Tensor *x, const framework::Tensor *y,
                   framework::Tensor *z, const DeviceContext &ctx, Functor func,
                   const bool is_xsize_larger = true)
      : x_(x->data<T>()),
        y_(y->data<T>()),
        z_(z->mutable_data<OutType>(ctx.GetPlace())),
        nx_(x->numel()),
        ctx_(ctx),
        func_(func),
        is_xsize_larger_(is_xsize_larger) {
    if (is_xsize_larger_ == false) {
      nx_ = y->numel();
    }
  }

  inline void Run() const {
    platform::Transform<DeviceContext> trans;
    trans(ctx_, x_, x_ + nx_, y_, z_, func_);
  }

  inline void RunRowWise(int n, int pre) const {
    platform::Transform<DeviceContext> trans;
    if (is_xsize_larger_) {
      trans(ctx_, x_, x_ + nx_,
            math::RowwiseTransformIterator<T, DeviceContext>(y_, n), z_, func_);
    } else {
      trans(ctx_, y_, y_ + nx_,
            math::RowwiseTransformIterator<T, DeviceContext>(x_, n), z_, func_);
    }
  }

  inline void RunMidWise(int n, int pre, int post) const {
    platform::Transform<DeviceContext> trans;
    if (is_xsize_larger_) {
      trans(ctx_, x_, x_ + nx_,
            math::MidWiseTransformIterator<T, DeviceContext>(y_, n, post), z_, func_);
    } else {
      trans(ctx_, y_, y_ + nx_,
            math::MidWiseTransformIterator<T, DeviceContext>(x_, n, post), z_, func_);
    }
  }

 private:
  const T *x_;
  const T *y_;
  OutType *z_;
  int64_t nx_;
  const DeviceContext &ctx_;
  Functor func_;
  bool is_xsize_larger_;
};

const size_t SHARE_NUM =  2;

template <typename DeviceContext, typename T>
class MpcElementwiseAddKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override{
        auto *in_x_t = ctx.Input<framework::LoDTensor>("X");
        auto *in_y_t = ctx.Input<framework::LoDTensor>("Y");
        auto *out_t = ctx.Output<framework::LoDTensor>("Out");

        int axis = ctx.Attr<int>("axis");

        auto out = out_t->mutable_data<T>(ctx.GetPlace());

        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->add(in_x_t, in_y_t, out_t, axis);
  }
};

template <typename DeviceContext, typename T>
class MpcElementwiseAddGradKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override {
        auto *in_x_t = ctx.Input<framework::LoDTensor>("X");
        auto *in_y_t = ctx.Input<framework::LoDTensor>("Y");
        auto *dout = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
        auto *dx = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
        auto *dy = ctx.Output<framework::LoDTensor>(framework::GradVarName("Y"));
        int axis = ctx.Attr<int>("axis");
        auto dout_data = dout->data<T>();

        if (dx) {
            auto dx_data = dx->mutable_data<T>(ctx.GetPlace());
            for (size_t i = 0; i < dout->numel(); i++) {
                dx_data[i] = dout_data[i];
            }
        }

        if (dy) {
            auto dy_data = dy->mutable_data<T>(ctx.GetPlace());
            if (in_x_t->dims().size() == in_y_t->dims().size()) {
                for (size_t i = 0; i < dout->numel(); i++) {
                    dy_data[i] = dout_data[i];
                }
            } else {
                auto x_dims = in_x_t->dims();
                auto y_dims = in_y_t->dims();

                axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);
                PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                     "Axis should be in range [0, x_dims)");

                int pre, n, post;
                math::GetMidDims get_mid_dims;
                get_mid_dims(x_dims, y_dims, axis, &pre, &n, &post);

                std::fill(dy_data, dy_data + dy->numel(), static_cast<T>(0));

                for (size_t i = 0; i < SHARE_NUM; ++i) {
                    int y_offset = i * n;
                    for (size_t j = 0; j < pre; ++j) {
                        for (size_t k = 0; k < n; ++k) {
                            for (size_t m = 0; m < post; ++m) {
                                int out_offset = i * pre * n * post + j * n * post + k * post + m;
                                dy_data[k + y_offset] += dout_data[out_offset];
                            }
                        }
                    }
                 }
            }
        }
    }
};

}  // namespace operators
}  // namespace paddle

