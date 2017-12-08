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

/// \brief Template for rendering  operations source code
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
  void OpClass(const Type& op_class) {
    op_class_ = op_class;
  }

  /// \brief Define an input to the operation
  void AddInput(const Variable& input);

  /// \brief Define an output of the operation
  void AddOutput(const Variable& output);

  /// \brief Define an attribute to the operation
  ///
  /// If the attribute has a default value when absent, it should be flagged
  /// as optional
  void AddAttribute(const Variable& attr, bool optional) {
    AddVariableiable(attr, optional ? &opt_attrs_ : &attrs_);
  }

  /// \brief Define an attribute providing a type for an operation
  void AddTypeAttribute(const Variable& attr) {
    AddVariableiable(attr, &attrs_);
    imports_.insert(Type::Enum("DataType", "org.tensorflow"));
  }

 private:
  enum RenderMode {
    DEFAULT,
    SINGLE_OUTPUT,
    SINGLE_LIST_OUTPUT
  };
  const string op_name_;
  Type op_class_;
  std::set<Type> imports_;
  std::vector<Variable> inputs_;
  std::vector<Variable> attrs_;
  std::vector<Variable> opt_attrs_;
  std::vector<Variable> outputs_;

  void AddVariableiable(const Variable& var, std::vector<Variable>* list) {
    CollectImports(var.type());
    list->push_back(var);
  }
  void CollectImports(const Type& type);
  void Render(SourceWriter* src_writer);
  void RenderOptionsClass(ClassWriter* op_writer);
  void RenderFactoryMethod(ClassWriter* op_writer);
  void RenderMethods(ClassWriter* op_writer, RenderMode mode,
      const Type& single_type);
  void RenderConstructor(ClassWriter* op_writer);
};

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_OP_TEMPLATE_H_
