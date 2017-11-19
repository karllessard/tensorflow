/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_JAVA_SRC_GEN_CC_OP_TEMPLATE_H_
#define TENSORFLOW_JAVA_SRC_GEN_CC_OP_TEMPLATE_H_

#include <string>
#include <set>
#include <vector>
#include <map>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"
#include "tensorflow/java/src/gen/cc/java_writer.h"

namespace tensorflow {
namespace java {

/// \brief Template for rendering Java operations source code
///
/// The ops generator collects operation data from the protobuf definitions
/// and store it to this template. Once all data is collected, the template
/// is rendered to the selected destination (a file or, for tests, in memory).
class OpTemplate {
 public:
  OpTemplate(const string& op_name);
  virtual ~OpTemplate() {}

  /// \brief Render this template to a file
  ///
  /// The file will be named and located by the op class definition.
  void RenderToFile(const string& root_dir, Env* env = Env::Default());

  /// \brief Render this templte to a memory buffer
  void RenderToBuffer(string* buffer);

  /// \brief Define the operation class to render
  ///
  /// Note that no supertype should be set explicitly in the class definition
  /// since they will (and must be) handle by the template itself.
  void OpClass(const JavaType& op_class) {
    op_class_ = op_class;
  }

  /// \brief Define an input to the operation
  void AddInput(const JavaVar& input);

  /// \brief Define an output of the operation
  void AddOutput(const JavaVar& output);

  /// \brief Define an attribute to the operation
  ///
  /// If the attribute has a default value when absent, it should be flagged
  /// as optional
  void AddAttribute(const JavaVar& attr, bool optional) {
    AddVariable(attr, optional ? &opt_attrs_ : &attrs_);
  }

  /// \brief Define an attribute providing a type for an operation
  void AddTypeAttribute(const JavaVar& attr) {
    AddVariable(attr, &attrs_);
    imports_.insert(Java::Enum("DataType", "org.tensorflow"));
  }

 private:
  enum RenderMode {
    DEFAULT,
    SINGLE_OUTPUT,
    SINGLE_LIST_OUTPUT
  };
  const string op_name_;
  JavaType op_class_;
  std::set<JavaType> imports_;
  std::vector<JavaVar> inputs_;
  std::vector<JavaVar> attrs_;
  std::vector<JavaVar> opt_attrs_;
  std::vector<JavaVar> outputs_;

  void AddVariable(const JavaVar& var, std::vector<JavaVar>* list) {
    CollectImports(var.type());
    list->push_back(var);
  }
  void CollectImports(const JavaType& type);
  void Render(SourceWriter* src_writer);
  void RenderOptionsClass(JavaClassWriter* op_writer);
  void RenderFactoryMethod(JavaClassWriter* op_writer);
  void RenderMethods(JavaClassWriter* op_writer, RenderMode mode,
      const JavaType& single_type);
  void RenderConstructor(JavaClassWriter* op_writer);
};

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_OP_TEMPLATE_H_
