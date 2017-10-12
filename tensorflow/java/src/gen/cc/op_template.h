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

#include <list>
#include <set>
#include <string>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/java/src/gen/cc/java_writer.h"

namespace tensorflow {

class OpTemplate {
 public:
  enum RenderMode {
    DEFAULT,
    SINGLE_OUTPUT,
    SINGLE_LIST_OUTPUT
  };
  OpTemplate(const string& op_name, const string& op_group);
  virtual ~OpTemplate() {}

  void Render(SourceOutputStream* stream);

  void OpClass(const JavaClass& class_tmpl) {
    this->op_class = class_tmpl;
  }
  void AddInput(const JavaVariable& input) {
    AddVariable(input, &inputs);
  }
  void AddAttribute(const JavaVariable& attr, bool optional) {
    AddVariable(attr, optional ? &opt_attrs : &attrs);
  }
  void AddOutput(const JavaVariable& output) {
    AddVariable(output, &outputs);
    if (IsList(output)) {
      imports.insert("java.util.Arrays");
    }
  }

 private:
  const string op_name;
  const string op_group;
  JavaClass op_class;
  std::set<string> imports;
  std::list<JavaVariable> inputs;
  std::list<JavaVariable> attrs;
  std::list<JavaVariable> opt_attrs;
  std::list<JavaVariable> outputs;

  RenderMode SelectRenderMode();
  void RenderOptionsClass(JavaClassWriter* op_writer);
  void RenderFactoryMethod(JavaClassWriter* op_writer, bool with_options);
  void RenderMethods(JavaClassWriter* op_writer, RenderMode mode);
  void RenderConstructor(JavaClassWriter* op_writer);
  void CollectImports(const JavaType& type);
  void AddVariable(const JavaVariable& var, std::list<JavaVariable>* list) {
    CollectImports(var.type());
    list->push_back(var);
  }
  static bool IsList(const JavaVariable& var) {
    return var.type().name() == "List" || var.type().name() == "Iterable";
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_OP_TEMPLATE_H_
