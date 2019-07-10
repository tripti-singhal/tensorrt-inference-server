// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/core/ensemble_utils.h"

#include <set>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config_utils.h"

namespace nvidia { namespace inferenceserver {

namespace {

// [TODO] rethink different flags
void
UpdateDownstreamState(
    std::set<DependencyNode*>* start_nodes,
    std::set<DependencyNode*>* updated_nodes)
{
  // Mark downstream nodes as unchecked recursively
  for (auto& dependent : *start_nodes) {
    dependent->checked_ = false;
    UpdateDownstreamState(&dependent->downstream_nodes_, updated_nodes);
    updated_nodes->emplace(dependent);
  }
}

// [TODO] verify what has been checked in ValidateModelConfig()
Status
ValidateTensorMapping(
    const std::string& ensemble, const ModelEnsembling::Step& step,
    const ModelConfig& model_config,
    std::unordered_map<std::string, TensorNode>* ensemble_tensors)
{
  // Check all inputs are mapped and no mapping to invalid inputs
  std::set<std::string> input_names;
  for (const auto& model_input : model_config.input()) {
    input_names.insert(model_input.name());
  }
  for (const auto& input_map : step.input_map()) {
    if (input_names.find(input_map.first) == input_names.end()) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "in ensemble " + ensemble + ", ensemble tensor " + input_map.second +
              " is mapping to non-existing input " + input_map.first +
              " in model " + step.model_name());
    }
  }
  for (const auto& model_input : model_config.input()) {
    size_t mapped_cnt = 0;
    for (const auto& input_map : step.input_map()) {
      if (model_input.name() == input_map.first) {
        TensorNode model_tensor(
            step.model_name(), model_input.data_type(), model_input.dims());
        auto it = ensemble_tensors->find(input_map.second);
        if (it != ensemble_tensors->end()) {
          RETURN_IF_ERROR(ValidateTensorConsistency(
              it->second, model_tensor,
              "in ensemble " + ensemble + ", ensemble tensor " +
                  input_map.second + ": "));
        } else {
          ensemble_tensors->emplace(
              std::make_pair(input_map.second, model_tensor));
        }
        mapped_cnt++;
      }
    }
    if (mapped_cnt == 0) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "in ensemble " + ensemble + ", input " + model_input.name() +
              " in model " + model_config.name() +
              " is not mapped to any ensemble tensors");
    } else if (mapped_cnt > 1) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "in ensemble " + ensemble + ", input " + model_input.name() +
              " in model " + model_config.name() +
              " is mapped to multiple ensemble tensors");
    }
  }

  // Check no multiple mappings to same ensemble tensor
  // and no mapping from invalid outputs
  std::set<std::string> output_names;
  for (const auto& model_output : model_config.output()) {
    output_names.insert(model_output.name());
  }
  for (const auto& output_map : step.output_map()) {
    if (output_names.find(output_map.first) == output_names.end()) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "in ensemble " + ensemble + ", ensemble tensor " + output_map.second +
              " is mapped from non-existing output " + output_map.first +
              " in model " + step.model_name());
    }
  }
  for (const auto& output_map : step.output_map()) {
    size_t mapped_cnt = 0;
    for (const auto& model_output : model_config.output()) {
      if (model_output.name() == output_map.first) {
        TensorNode model_tensor(
            step.model_name(), model_output.data_type(), model_output.dims());
        auto it = ensemble_tensors->find(output_map.second);
        if (it != ensemble_tensors->end()) {
          RETURN_IF_ERROR(ValidateTensorConsistency(
              it->second, model_tensor,
              "in ensemble " + ensemble + ", ensemble tensor " +
                  output_map.second + ": "));
        } else {
          ensemble_tensors->emplace(
              std::make_pair(output_map.second, model_tensor));
        }
        mapped_cnt++;
      }
    }
    if (mapped_cnt > 1) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "in ensemble " + ensemble + ", multiple outputs in model " +
              model_config.name() + " are mapped to the same ensemble tensor " +
              output_map.second);
    }
  }

  // link ensemble tensors
  for (const auto& output_map : step.output_map()) {
    auto& node = ensemble_tensors->find(output_map.second)->second;
    for (const auto& input_map : step.input_map()) {
      auto& prev_node = ensemble_tensors->find(input_map.second)->second;
      node.prev_nodes_.push_back(&prev_node);
      prev_node.next_nodes_.push_back(&node);
    }
  }
  return Status::Success;
}

}  // namespace

Status
ValidateTensorConsistency(
    const TensorNode& lhs, const TensorNode& rhs, const std::string& message)
{
  if (lhs.type_ != rhs.type_) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        message + "inconsistent data type: " + std::to_string(lhs.type_) +
            " is inferred from model " + lhs.model_name_ + " while " +
            std::to_string(rhs.type_) + " is inferred from model " +
            rhs.model_name_);
  }

  bool consistent = (lhs.dims_.size() == rhs.dims_.size());
  if (consistent) {
    for (int i = 0; i < lhs.dims_.size(); i++) {
      if (lhs.dims_[i] != rhs.dims_[i]) {
        consistent = false;
        break;
      }
    }
  }
  if (!consistent) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        message + "inconsistent shape: " + DimsListToString(lhs.dims_) +
            " is inferred from model " + lhs.model_name_ + " while " +
            DimsListToString(rhs.dims_) + " is inferred from model " +
            rhs.model_name_);
  }

  return Status::Success;
}

// [TODO] break this up
Status
ValidateEnsembleConfig(
    const std::string& ensemble,
    std::unordered_map<std::string, bool>* ensembles,
    std::deque<std::string>* ensemble_dependency,
    std::unordered_map<std::string, std::unique_ptr<DependencyNode>>*
        dependency_graph,
    std::unordered_map<std::string, std::unique_ptr<DependencyNode>>*
        missing_nodes)
{
  std::unordered_map<std::string, TensorNode> ensemble_tensors;

  const auto& ensemble_config =
      dependency_graph->find(ensemble)->second->model_config_;

  for (const auto& input : ensemble_config.input()) {
    TensorNode input_node(ensemble, input.data_type(), input.dims());
    ensemble_tensors.emplace(std::make_pair(input.name(), input_node));
  }
  for (const auto& output : ensemble_config.output()) {
    TensorNode output_node(ensemble, output.data_type(), output.dims());
    ensemble_tensors.emplace(std::make_pair(output.name(), output_node));
  }

  auto ensemble_node = dependency_graph->find(ensemble)->second.get();
  // [TODO] only updated ensembles? No, still need to check upstream anyway
  // check model config again
  ensemble_node->upstream_nodes_.clear();
  for (const auto& step : ensemble_config.ensemble_scheduling().step()) {
    const auto& model_name = step.model_name();
    auto dit = dependency_graph->find(model_name);
    if (dit == dependency_graph->end()) {
      auto mit = missing_nodes->find(model_name);
      if (mit == missing_nodes->end()) {
        std::unique_ptr<DependencyNode> node(new DependencyNode(model_name));
        ensemble_node->missing_upstream_nodes_.emplace(node.get());
        mit = missing_nodes->emplace(model_name, std::move(node)).first;
      }
      mit->second->downstream_nodes_.emplace(ensemble_node);
      ensemble_node->upstream_nodes_.emplace(
          mit->second.get(), step.model_version());
      ensemble_node->status_ = Status(
          RequestStatusCode::INVALID_ARG,
          "ensemble " + ensemble + " contains model " + model_name +
              " which is not in the available models");
      // continue to complete the edges of the graph
      continue;
    } else {
      dit->second->downstream_nodes_.emplace(ensemble_node);
      ensemble_node->upstream_nodes_.emplace(
          dit->second.get(), step.model_version());
    }
  }

  if (ensemble_node->status_.IsOk()) {
    for (const auto& step : ensemble_config.ensemble_scheduling().step()) {
      const auto& model_name = step.model_name();
      ModelConfig model_config;
      for (auto& node : ensemble_node->upstream_nodes_) {
        if (model_name == node.first->model_name_) {
          model_config = node.first->model_config_;
          break;
        }
      }

      if (model_config.max_batch_size() < ensemble_config.max_batch_size()) {
        ensemble_node->status_ = Status(
            RequestStatusCode::INVALID_ARG,
            "ensemble " + ensemble + " allows maximum batch size " +
                std::to_string(ensemble_config.max_batch_size()) +
                ", but it contains model " + model_name +
                " which only allows maximum batch size to be " +
                std::to_string(model_config.max_batch_size()));
        break;
      }

      if (model_config.has_ensemble_scheduling()) {
        bool found = false;
        for (const auto& name : *ensemble_dependency) {
          if (name == model_name) {
            found = true;
            break;
          }
        }
        if (found) {
          ensemble_node->status_ = Status(
              RequestStatusCode::INVALID_ARG,
              "circular dependency between ensembles: " + model_name +
                  " -> ... -> " + ensemble + " -> " + model_name);
          break;
        }

        if ((ensembles->find(model_name))->second == false) {
          ensemble_dependency->push_back(ensemble);
          Status status = ValidateEnsembleConfig(
              model_name, ensembles, ensemble_dependency, dependency_graph,
              missing_nodes);
          ensemble_dependency->pop_back();
          if (!status.IsOk()) {
            ensemble_node->status_ = Status(
                RequestStatusCode::INVALID_ARG,
                "ensemble " + ensemble + " depends on " + model_name +
                    " which contains invalid model config");
            break;
          }
        }
      }

      ensemble_node->status_ = ValidateTensorMapping(
          ensemble, step, model_config, &ensemble_tensors);
      if (!ensemble_node->status_.IsOk()) {
        break;
      }
    }
  }

  (ensembles->find(ensemble))->second = true;
  return ensemble_node->status_;
}

Status
UpdateDependencyGraph(
    ModelRepositoryManager* manager, const std::set<std::string>& added,
    const std::set<std::string>& deleted, const std::set<std::string>& modified,
    std::unordered_map<std::string, std::unique_ptr<DependencyNode>>*
        dependency_graph,
    std::unordered_map<std::string, std::unique_ptr<DependencyNode>>*
        missing_nodes)
{
  // update dependency graph
  // keep track of dependents as changes are going to affect downstream instead
  // of upstream

  // deleted, drop from dependency_graph, add to missing_nodes if downstreams is
  // not empty affected_nodes are all ensembles as only ensembles are depending
  // on other models
  std::set<DependencyNode*> affected_nodes;
  std::set<DependencyNode*> updated_nodes;
  for (const auto& model_name : deleted) {
    auto it = dependency_graph->find(model_name);
    if (it != dependency_graph->end()) {
      if (!it->second->downstream_nodes_.empty()) {
        UpdateDownstreamState(&it->second->downstream_nodes_, &affected_nodes);
        // mark this node as missing upstream node in its downstream nodes
        for (auto& dependent : it->second->downstream_nodes_) {
          dependent->missing_upstream_nodes_.emplace(it->second.get());
        }

        // remove this node from its upstream node
        for (auto& upstream : it->second->upstream_nodes_) {
          upstream.first->downstream_nodes_.erase(it->second.get());
        }
        it->second->upstream_nodes_.clear();

        (*missing_nodes)[model_name] = std::move(it->second);
      }
      // Make sure deleted node will not be in affected nodes
      affected_nodes.erase(it->second.get());
      dependency_graph->erase(it);
    }
  }

  // modified, invalidate (uncheck) all downstreams
  for (const auto& model_name : modified) {
    auto it = dependency_graph->find(model_name);
    if (it != dependency_graph->end()) {
      UpdateDownstreamState(&it->second->downstream_nodes_, &affected_nodes);
      manager->GetModelConfig(model_name, &it->second->model_config_);
      // remove this node from its upstream node
      for (auto& upstream : it->second->upstream_nodes_) {
        upstream.first->downstream_nodes_.erase(it->second.get());
      }
      it->second->upstream_nodes_.clear();
      it->second->checked_ = false;
      updated_nodes.emplace(it->second.get());
    }
  }

  // added, add to dependency_graph, if in missing_node, invalidate (uncheck)
  // and associate all downstreams, remove from missing_node
  for (const auto& model_name : added) {
    std::unique_ptr<DependencyNode> added_node;
    auto it = missing_nodes->find(model_name);
    if (it != missing_nodes->end()) {
      UpdateDownstreamState(&it->second->downstream_nodes_, &affected_nodes);
      // remove this node from missing upstream node in its downstream nodes
      for (auto& dependent : it->second->downstream_nodes_) {
        dependent->missing_upstream_nodes_.erase(it->second.get());
      }

      it->second->checked_ = false;
      added_node = std::move(it->second);
      //(*dependency_graph)[model_name] = std::move(it->second);
      missing_nodes->erase(it);
    } else {
      // Right now, nothing is going to be filled until validation
      added_node.reset(new DependencyNode(model_name));
    }
    manager->GetModelConfig(model_name, &added_node->model_config_);
    updated_nodes.emplace(added_node.get());
    dependency_graph->emplace(model_name, std::move(added_node));
  }

  // [TODO] collect modified configs and ValidateEnsembleConfig() here
  ValidateEnsembleConfig(
      updated_nodes, affected_nodes, dependency_graph, missing_nodes);
  // [TODO] return <valid, invalid> pair?
  return Status::Success;
}

Status
ValidateEnsembleConfig(
    const std::set<DependencyNode*>& updated_nodes,
    const std::set<DependencyNode*>& affected_nodes,
    std::unordered_map<std::string, std::unique_ptr<DependencyNode>>*
        dependency_graph,
    std::unordered_map<std::string, std::unique_ptr<DependencyNode>>*
        missing_nodes)
{
  std::unordered_map<std::string, bool> ensembles;

  for (const auto& node : affected_nodes) {
    ensembles.emplace(std::make_pair(node->model_name_, false));
  }
  ModelConfig model_config;
  for (auto& node : updated_nodes) {
    if (node->model_config_.has_ensemble_scheduling()) {
      ensembles.emplace(std::make_pair(node->model_name_, false));
    } else {
      // non-ensemble model config should have been checked beforehand.
      node->checked_ = true;
    }
    // assume all nodes are valid
    node->status_ = Status::Success;
  }

  std::deque<std::string> ensemble_dependency;
  for (const auto& pair : ensembles) {
    if (pair.second) {
      continue;
    }
    RETURN_IF_ERROR(ValidateEnsembleConfig(
        pair.first, &ensembles, &ensemble_dependency, dependency_graph,
        missing_nodes));
  }

  return Status::Success;
}

std::pair<std::set<std::string>, std::set<std::string>>
ModelsToLoad(
    std::unordered_map<std::string, std::set<int64_t>> loaded_models,
    std::unordered_map<std::string, std::unique_ptr<DependencyNode>>*
        dependency_graph)
{
  std::pair<std::set<std::string>, std::set<std::string>> res;
  // first call to this function
  if (loaded_models.empty()) {
    for (auto& pair : (*dependency_graph)) {
      auto& node = pair.second;
      // only care about nodes that are affected by the update
      if (!node->checked_) {
        // the node failed on validation
        if (!node->status_.IsOk()) {
          res.second.emplace(node->model_name_);
        } else {
          bool node_ready = true;
          bool node_valid = true;
          for (auto& upstream : node->upstream_nodes_) {
            if (!upstream.first->checked_) {
              node_ready = false;
              break;
            }
            if (!upstream.first->status_.IsOk()) {
              node_valid = false;
              break;
            }
            // check if the required version of upstream is loaded
            if (upstream.first->loaded_versions_.empty()) {
              node_valid = false;
              break;
            } else if (upstream.second != -1) {
              auto it = upstream.first->loaded_versions_.find(upstream.second);
              if (it == upstream.first->loaded_versions_.end()) {
                node_valid = false;
                break;
              }
            }
          }
          if (node_ready) {
            if (node_valid) {
              res.first.emplace(node->model_name_);
            } else {
              res.second.emplace(node->model_name_);
            }
          }
        }
      }
    }
  } else {
    // check downstream of the loaded model is sufficient
    for (const auto& model : loaded_models) {
      // [TODO] return DependencyNode directly? maybe after DependencyGraph is
      // included in ModelRepositoryManager
      auto it = dependency_graph->find(model.first);
      // update loaded version
      it->second->loaded_versions_ = model.second;
      it->second->checked_ = true;
      for (auto& node : it->second->downstream_nodes_) {
        // only care about nodes that are affected by the update
        if (!node->checked_) {
          // the node failed on validation
          if (!node->status_.IsOk()) {
            res.second.emplace(node->model_name_);
          } else {
            bool node_ready = true;
            bool node_valid = true;
            for (auto& upstream : node->upstream_nodes_) {
              if (!upstream.first->checked_) {
                node_ready = false;
                break;
              }
              if (!upstream.first->status_.IsOk()) {
                node_valid = false;
                break;
              }
              // check if the required version of upstream is loaded
              if (upstream.first->loaded_versions_.empty()) {
                node_valid = false;
                break;
              } else if (upstream.second != -1) {
                auto it =
                    upstream.first->loaded_versions_.find(upstream.second);
                if (it == upstream.first->loaded_versions_.end()) {
                  node_valid = false;
                  break;
                }
              }
            }
            if (node_ready) {
              if (node_valid) {
                res.first.emplace(node->model_name_);
              } else {
                res.second.emplace(node->model_name_);
              }
            }
          }
        }
      }
    }
  }
  return res;
}

}}  // namespace nvidia::inferenceserver
