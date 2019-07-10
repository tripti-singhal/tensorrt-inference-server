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
#pragma once

#include <deque>
#include <unordered_map>
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_repository_manager.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

/// A basic unit in ensemble graph that records the data type and shape
/// of the ensemble tensor and which model they are inferred from.
struct TensorNode {
  TensorNode(const std::string& model_name, const DataType& type, const DimsList& dims)
      : model_name_(model_name), type_(type), dims_(dims), ready_(false)
  {
  }

  std::string model_name_;
  DataType type_;
  DimsList dims_;
  bool ready_;
  std::vector<TensorNode*> prev_nodes_;
  std::vector<TensorNode*> next_nodes_;
};

/// A basic unit in dependency graph that records the models seen by the model repository manager.
struct DependencyNode {
  DependencyNode(const std::string& model_name)
      : model_name_(model_name), status_(Status::Success), checked_(false)
  {
  }

  std::string model_name_;
  Status status_;
  bool checked_;
  ModelConfig model_config_;
  std::set<DependencyNode*> missing_upstream_nodes_;
  std::set<DependencyNode*> upstream_nodes_;
  std::set<DependencyNode*> downstream_nodes_;
};

/// Validate if the data type and the shape of two TensorNode object are
/// consistent.
/// \param lhs One of the TensorNode object to be validated.
/// \param rhs Another TensorNode object to be validated.
/// \param message Extra message included in the front of error message
/// if error status is non-OK.
/// \return The error status. A non-OK status indicates the TensorNode objects
/// are not consistent.
Status ValidateTensorConsistency(
    const TensorNode& lhs, const TensorNode& rhs, const std::string& message);

/// [TODO] add / update all docs
Status
UpdateDependencyGraph(
    ModelRepositoryManager* manager,
    const std::set<std::string>& added, const std::set<std::string>& deleted,
    const std::set<std::string>& modified,
    std::unordered_map<std::string, std::unique_ptr<DependencyNode>>* dependency_graph,
    std::unordered_map<std::string, std::unique_ptr<DependencyNode>>* missing_nodes);

/// Validate that the ensembles are specified correctly. Assuming that the
/// inputs and outputs specified in all model configurations are accurate.
/// \param config_map Map from model name to model configuration to validate.
/// It contains the model configurations of the ensembles and the models
/// in the ensembles.
/// \return The error status. A non-OK status indicates the configuration
/// is not valid.
Status ValidateEnsembleConfig(
    const std::set<DependencyNode*>& updated_nodes,
    const std::set<DependencyNode*>& affected_nodes,
    std::unordered_map<std::string, std::unique_ptr<DependencyNode>>* dependency_graph,
    std::unordered_map<std::string, std::unique_ptr<DependencyNode>>* missing_nodes);

/// Validate that the ensembles are specified correctly. Assuming that the
/// inputs and outputs specified in all model configurations are accurate.
/// \param ensemble The name of the ensemble to validate. Its model
/// configuration must be included in 'config_map'.
/// \param config_map Map from model name to model configuration. It contains
/// the model configurations of the ensembles and the models in the ensembles.
/// \param invalid_model_names Names of models that are in 'config_map' but are
/// invalid due to incomplete specification in inputs and/or outputs.
/// \param ensembles Map from ensemble name that is in 'config_map' to boolean
/// which indicate if the ensemble has been validated.
/// \param ensemble_dependency A queue that shows the inclusion hierarchy that
/// 'ensemble' is in.
/// \return The error status. A non-OK status indicates the configuration
/// is not valid.
Status ValidateEnsembleConfig(
    const std::string& ensemble,
    std::unordered_map<std::string, bool>* ensembles,
    std::deque<std::string>* ensemble_dependency,
    std::unordered_map<std::string, std::unique_ptr<DependencyNode>>* dependency_graph,
    std::unordered_map<std::string, std::unique_ptr<DependencyNode>>* missing_nodes);

}}  // namespace nvidia::inferenceserver
