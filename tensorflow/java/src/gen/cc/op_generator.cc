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

#include <string>
#include <map>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/java/src/gen/cc/op_generator.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"
#include "tensorflow/java/src/gen/cc/op_template.h"
#include "tensorflow/java/src/gen/cc/op_type_resolver.h"

namespace tensorflow {
namespace java {
namespace {

string SnakeToCamelCase(const string& str, bool upper = false) {
  string result;
  bool cap = upper;
  for (string::const_iterator it = str.begin(); it != str.end(); ++it) {
    const char c = *it;
    if (c == '_') {
      cap = true;
    } else if (cap) {
      result += toupper(c);
      cap = false;
    } else {
      result += c;
    }
  }
  return result;
}

inline bool IsParamOf(const JavaType& type, const JavaType& clazz) {
  return std::find(clazz.params().begin(), clazz.params().end(), type)
      != clazz.params().end();
}

inline bool IsInternal(const OpDef& op) {
  return op.name()[0] == '_';
}

}  // namespace

OpGenerator::OpGenerator() : env(Env::Default()) {}

OpGenerator::~OpGenerator() {}

Status OpGenerator::Run(const OpList& ops, const string& lib_name,
    const string& base_package, const string& output_dir) {
  const string op_group = SnakeToCamelCase(lib_name);
  LOG(INFO) << "Generating Java wrappers for '" << lib_name << "' operations";
  for (const auto& op : ops.op()) {
    if (!IsInternal(op)  // skip internal ops
        && GenerateOp(op, op_group, base_package, output_dir) != Status::OK()) {
      LOG(ERROR) << "Fail to generate Java wrapper for operation \""
          << op.name() << "\"";
    }
  }
  return Status::OK();
}

Status OpGenerator::GenerateOp(const OpDef& op, const string& op_group,
    const string& base_package, const string& output_dir) {
  OpTypeResolver type_resolver;
  OpTemplate tmpl(op.name());
  const string package = base_package + '.' + str_util::Lowercase(op_group);
  JavaType op_class = Java::Class(op.name(), package);

  for (const auto& input : op.input_arg()) {
    const string input_name = SnakeToCamelCase(input.name());
    const ResolvedType type = type_resolver.InputType(input, op);
    tmpl.AddInput(Java::Var(input_name, type.var));
  }
  for (const auto& output : op.output_arg()) {
    const string output_name = SnakeToCamelCase(output.name());
    const ResolvedType type = type_resolver.OutputType(output, op);
    if (Java::IsGeneric(type.tensor) && !IsParamOf(type.tensor, op_class)) {
      op_class.param(type.tensor);
    }
    tmpl.AddOutput(Java::Var(output_name, type.var), type.is_new_generic);
  }
  for (const auto& attr : op.attr()) {
    if (!type_resolver.IsInferred(attr)) {  // skip type attributes
      const string attr_name = SnakeToCamelCase(attr.name());
      const JavaType type = type_resolver.AttrType(attr);
      tmpl.AddAttribute(Java::Var(attr_name, type), attr.has_default_value());
    }
  }
  tmpl.OpClass(op_class);
  tmpl.RenderToFile(output_dir, env);

  return Status::OK();
}


}  // namespace java
}  // namespace tensorflow
