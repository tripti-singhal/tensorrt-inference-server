// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <unistd.h>
#include <chrono>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "src/core/api.pb.h"
#include "src/core/server_status.pb.h"
#include "src/core/trtserver.h"
#include "src/servers/common.h"

namespace ni = nvidia::inferenceserver;

namespace {

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-r [model repository absolute path]" << std::endl;

  exit(1);
}

TRTSERVER_Error*
MemoryAlloc(
    void** buffer, size_t byte_size, TRTSERVER_MemoryAllocator_Region region,
    int64_t region_id)
{
  FAIL("MemoryAlloc: NYI");
  return nullptr;  // Success
}

TRTSERVER_Error*
MemoryDelete(
    void* buffer, size_t byte_size, TRTSERVER_MemoryAllocator_Region region,
    int64_t region_id)
{
  FAIL("MemoryDelete: NYI");
  return nullptr;  // Success
}

void
InferComplete(
    TRTSERVER_Server* server, TRTSERVER_InferenceResponse* response,
    void* userp)
{
  std::promise<TRTSERVER_InferenceResponse*>* p =
      reinterpret_cast<std::promise<TRTSERVER_InferenceResponse*>*>(userp);
  p->set_value(response);
  delete p;
}

}  // namespace

int
main(int argc, char** argv)
{
  std::string model_repository_path;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "r:")) != -1) {
    switch (opt) {
      case 'r':
        model_repository_path = optarg;
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  if (model_repository_path.empty()) {
    Usage(argv, "-r must be used to specify model repository path");
  }

  // Create the server...
  TRTSERVER_ServerOptions* server_options = nullptr;
  FAIL_IF_ERR(
      TRTSERVER_ServerOptionsNew(&server_options), "creating server options");
  FAIL_IF_ERR(
      TRTSERVER_ServerOptionsSetModelRepositoryPath(
          server_options, model_repository_path.c_str()),
      "setting model repository path");

  TRTSERVER_Server* server_ptr = nullptr;
  FAIL_IF_ERR(
      TRTSERVER_ServerNew(&server_ptr, server_options), "creating server");
  FAIL_IF_ERR(
      TRTSERVER_ServerOptionsDelete(server_options), "deleting server options");

  std::shared_ptr<TRTSERVER_Server> server(server_ptr, TRTSERVER_ServerDelete);

  // Wait until the server is both live and ready.
  size_t health_iters = 0;
  while (true) {
    bool live, ready;
    FAIL_IF_ERR(
        TRTSERVER_ServerIsLive(server.get(), &live),
        "unable to get server liveness");
    FAIL_IF_ERR(
        TRTSERVER_ServerIsReady(server.get(), &ready),
        "unable to get server readiness");
    std::cout << "Server Health: live " << live << ", ready " << ready
              << std::endl;
    if (live && ready) {
      break;
    }

    if (++health_iters >= 10) {
      FAIL("failed to find healthy inference server");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // Print status of the server.
  {
    TRTSERVER_Protobuf* server_status_protobuf;
    FAIL_IF_ERR(
        TRTSERVER_ServerStatus(server.get(), &server_status_protobuf),
        "unable to get server status protobuf");
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(
        TRTSERVER_ProtobufSerialize(
            server_status_protobuf, &buffer, &byte_size),
        "unable to serialize server status protobuf");

    ni::ServerStatus server_status;
    if (!server_status.ParseFromArray(buffer, byte_size)) {
      FAIL("error: failed to parse server status");
    }

    std::cout << "Server Status:" << std::endl;
    std::cout << server_status.DebugString() << std::endl;

    FAIL_IF_ERR(
        TRTSERVER_ProtobufDelete(server_status_protobuf),
        "deleting status protobuf");
  }

  // Wait for the simple model to become available.
  while (true) {
    TRTSERVER_Protobuf* model_status_protobuf;
    FAIL_IF_ERR(
        TRTSERVER_ServerModelStatus(
            server.get(), &model_status_protobuf, "simple"),
        "unable to get model status protobuf");
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(
        TRTSERVER_ProtobufSerialize(model_status_protobuf, &buffer, &byte_size),
        "unable to serialize model status protobuf");

    ni::ServerStatus model_status;
    if (!model_status.ParseFromArray(buffer, byte_size)) {
      FAIL("error: failed to parse model status");
    }

    auto itr = model_status.model_status().find("simple");
    if (itr == model_status.model_status().end()) {
      FAIL("unable to find status for model 'simple'");
    }

    auto vitr = itr->second.version_status().find(1);
    if (vitr == itr->second.version_status().end()) {
      FAIL("unable to find version 1 status for model 'simple'");
    }

    std::cout << "'simple' model is "
              << ni::ModelReadyState_Name(vitr->second.ready_state())
              << std::endl;
    if (vitr->second.ready_state() == ni::ModelReadyState::MODEL_READY) {
      break;
    }

    FAIL_IF_ERR(
        TRTSERVER_ProtobufDelete(model_status_protobuf),
        "deleting status protobuf");

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // Create the memory allocator...
  TRTSERVER_MemoryAllocator* allocator = nullptr;
  FAIL_IF_ERR(
      TRTSERVER_MemoryAllocatorNew(&allocator, MemoryAlloc, MemoryDelete),
      "creating memory allocator");

  // The inference request provides meta-data with an
  // InferRequestHeader and the actual data via a provider.
  const std::string model_name("simple");
  int64_t model_version = -1;  // latest

  ni::InferRequestHeader request_header;
  request_header.set_id(123);
  request_header.set_batch_size(1);

  auto input0 = request_header.add_input();
  input0->set_name("INPUT0");
  auto input1 = request_header.add_input();
  input1->set_name("INPUT1");

  auto output0 = request_header.add_output();
  output0->set_name("OUTPUT0");
  auto output1 = request_header.add_output();
  output1->set_name("OUTPUT1");

  std::string request_header_serialized;
  request_header.SerializeToString(&request_header_serialized);

  // Create the inference request provider which provides all the
  // input information needed for an inference.
  TRTSERVER_InferenceRequestProvider* request_provider = nullptr;
  FAIL_IF_ERR(
      TRTSERVER_InferenceRequestProviderNew(
          &request_provider, server.get(), model_name.c_str(), model_version,
          request_header_serialized.c_str(), request_header_serialized.size()),
      "creating inference request provider");

  // Create the data for the two input tensors. Initialize the first
  // to unique integers and the second to all ones.
  std::vector<int32_t> input0_data(16);
  std::vector<int32_t> input1_data(16);
  for (size_t i = 0; i < 16; ++i) {
    input0_data[i] = i;
    input1_data[i] = 1;
  }

  FAIL_IF_ERR(
      TRTSERVER_InferenceRequestProviderSetInputData(
          request_provider, input0->name().c_str(), &input0_data[0],
          input0_data.size() * sizeof(int32_t)),
      "assigning INPUT0 data");
  FAIL_IF_ERR(
      TRTSERVER_InferenceRequestProviderSetInputData(
          request_provider, input1->name().c_str(), &input1_data[0],
          input1_data.size() * sizeof(int32_t)),
      "assigning INPUT1 data");

  // Perform inference...
  auto p = new std::promise<TRTSERVER_InferenceResponse*>();
  std::future<TRTSERVER_InferenceResponse*> completed = p->get_future();

  FAIL_IF_ERR(
      TRTSERVER_ServerInferAsync(
          server.get(), request_provider,
          nullptr /* http_response_provider_hack */,
          nullptr /* grpc_response_provider_hack */, InferComplete,
          reinterpret_cast<void*>(p)),
      "running inference");

  // The request provider can be deleted immediately after the
  // ServerInferAsync call returns.
  FAIL_IF_ERR(
      TRTSERVER_InferenceRequestProviderDelete(request_provider),
      "deleting inference request provider");

  // Wait for the inference response and check the status.
  TRTSERVER_InferenceResponse* response = completed.get();
  FAIL_IF_ERR(TRTSERVER_InferenceResponseStatus(response), "response");

  // Print the response header metadata.
  {
    TRTSERVER_Protobuf* response_protobuf;
    FAIL_IF_ERR(
        TRTSERVER_InferenceResponseHeader(response, &response_protobuf),
        "unable to get response header protobuf");
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(
        TRTSERVER_ProtobufSerialize(response_protobuf, &buffer, &byte_size),
        "unable to serialize response header protobuf");

    ni::InferResponseHeader response_header;
    if (!response_header.ParseFromArray(buffer, byte_size)) {
      FAIL("error: failed to parse response header");
    }

    std::cout << "Model \"simple\" response header:" << std::endl;
    std::cout << response_header.DebugString() << std::endl;

    FAIL_IF_ERR(
        TRTSERVER_ProtobufDelete(response_protobuf),
        "deleting response protobuf");
  }

  // Check the output tensor values...
  const void* output0_content;
  size_t output0_byte_size;
  FAIL_IF_ERR(
      TRTSERVER_InferenceResponseOutputData(
          response, output0->name().c_str(), &output0_content,
          &output0_byte_size),
      "getting output0 result");
  if (output0_byte_size != (16 * sizeof(int32_t))) {
    FAIL(
        "unexpected output0 byte-size, expected " +
        std::to_string(16 * sizeof(int32_t)) + ", got " +
        std::to_string(output0_byte_size));
  }

  const void* output1_content;
  size_t output1_byte_size;
  FAIL_IF_ERR(
      TRTSERVER_InferenceResponseOutputData(
          response, output1->name().c_str(), &output1_content,
          &output1_byte_size),
      "getting output1 result");
  if (output1_byte_size != (16 * sizeof(int32_t))) {
    FAIL(
        "unexpected output1 byte-size, expected " +
        std::to_string(16 * sizeof(int32_t)) + ", got " +
        std::to_string(output1_byte_size));
  }

  const int32_t* output0_result =
      reinterpret_cast<const int32_t*>(output0_content);
  const int32_t* output1_result =
      reinterpret_cast<const int32_t*>(output1_content);

  for (size_t i = 0; i < 16; ++i) {
    std::cout << input0_data[i] << " + " << input1_data[i] << " = "
              << output0_result[i] << std::endl;
    std::cout << input0_data[i] << " - " << input1_data[i] << " = "
              << output1_result[i] << std::endl;

    if ((input0_data[i] + input1_data[i]) != output0_result[i]) {
      FAIL("incorrect sum in " + output0->name());
    }
    if ((input0_data[i] - input1_data[i]) != output1_result[i]) {
      FAIL("incorrect difference in " + output1->name());
    }
  }

  FAIL_IF_ERR(
      TRTSERVER_InferenceResponseDelete(response),
      "deleting inference response");
  FAIL_IF_ERR(
      TRTSERVER_MemoryAllocatorDelete(allocator), "deleting memory allocator");

  return 0;
}
