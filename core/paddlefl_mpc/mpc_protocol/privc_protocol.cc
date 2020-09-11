// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Description:
// an PrivC protocol impl, including combination of operator, network and circuit
// context

#include "privc_protocol.h"
#include "gloo/rendezvous/redis_store.h"
#include "mpc_protocol_factory.h"
#include "core/privc/ot.h"
#include "core/privc/triplet_generator.h"

namespace paddle {
namespace mpc {

void PrivCProtocol::init_with_store(
    const MpcConfig &config, std::shared_ptr<gloo::rendezvous::Store> store) {
  if (_is_initialized) {
    return;
  }

  PADDLE_ENFORCE_NOT_NULL(store);

  // read role, address and other info
  auto role = config.get_int(PrivCConfig::ROLE);
  PADDLE_ENFORCE_LT(role, 2, "Input role should be less than party_size(2).");

  auto local_addr =
      config.get(PrivCConfig::LOCAL_ADDR, PrivCConfig::LOCAL_ADDR_DEFAULT);
  auto net_server_addr = config.get(PrivCConfig::NET_SERVER_ADDR,
                                    PrivCConfig::NET_SERVER_ADDR_DEFAULT);
  auto net_server_port = config.get_int(PrivCConfig::NET_SERVER_PORT,
                                        PrivCConfig::NET_SERVER_PORT_DEFAULT);

  auto mesh_net = std::make_shared<MeshNetwork>(
      role, local_addr, 2 /* netsize */, "Paddle-mpc" /* key-prefix in store*/,
      store);
  mesh_net->init();

  _network = std::move(mesh_net);
  _circuit_ctx = std::make_shared<PrivCContext>(role, _network);
  _operators = std::make_shared<PrivCOperatorsImpl>();
  _is_initialized = true;

  // init tripletor and ot
  PADDLE_ENFORCE_NOT_NULL(_circuit_ctx);

  using TripletGenerator = privc::TripletGenerator<int64_t,
                                PRIVC_FIXED_POINTER_SCALING_FACTOR>;
  std::shared_ptr<TripletGenerator> tripletor
                  = std::make_shared<TripletGenerator>(_circuit_ctx);
  std::dynamic_pointer_cast<PrivCContext>(_circuit_ctx)
                  ->set_triplet_generator(tripletor);

  std::shared_ptr<privc::OT> ot = std::make_shared<privc::OT>(_circuit_ctx);
  ot->init();
  std::dynamic_pointer_cast<PrivCContext>(_circuit_ctx)
                  ->set_ot(ot);
}

std::shared_ptr<MpcOperators> PrivCProtocol::mpc_operators() {
  PADDLE_ENFORCE(_is_initialized, PROT_INIT_ERR);
  return _operators;
}

std::shared_ptr<AbstractNetwork> PrivCProtocol::network() {
  PADDLE_ENFORCE(_is_initialized, PROT_INIT_ERR);
  return _network;
}

std::shared_ptr<AbstractContext> PrivCProtocol::mpc_context() {
  PADDLE_ENFORCE(_is_initialized, PROT_INIT_ERR);
  return _circuit_ctx;
}

void PrivCProtocol::init(const MpcConfig &config) {
  if (_is_initialized) {
    return;
  }

  auto server_addr = config.get(PrivCConfig::NET_SERVER_ADDR,
                                PrivCConfig::NET_SERVER_ADDR_DEFAULT);
  auto server_port = config.get_int(PrivCConfig::NET_SERVER_PORT,
                                    PrivCConfig::NET_SERVER_PORT_DEFAULT);
  auto gloo_store =
      std::make_shared<gloo::rendezvous::RedisStore>(server_addr, server_port);

  init_with_store(config, gloo_store);
}

} // mpc
} // paddle
