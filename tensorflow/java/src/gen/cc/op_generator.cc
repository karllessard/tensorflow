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

void ParseDescription(const string& descr, JavaDoc* doc) {
  std::stringstream jdoc_descr;
  bool newline = false;
  for (std::string::const_iterator c = descr.cbegin(); c != descr.cend(); ++c) {
    if (*c == '\n') {
      if (newline) {
        jdoc_descr << "<p>\n";
      } else {
        jdoc_descr << '\n';
      }
      newline = true;
    } else {
      switch (*c) {
      case '`':
        jdoc_descr << "&#96;";
        break;
      case '/':
        jdoc_descr << "&#47;";
        break;
      default:
        jdoc_descr << *c;
        break;
      }
      newline = false;
    }
  }
  doc->descr(jdoc_descr.str());
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
  ParseDescription(op.description(), op_class.doc_ptr());

  for (const auto& input : op.input_arg()) {
    const string input_name = SnakeToCamelCase(input.name());
    const ResolvedType type = type_resolver.TypeOf(input, true);
    JavaType input_type = Java::Interface("Operand", "org.tensorflow")
        .param(type.dt);
    JavaVar input_var = Java::Var(input_name,
        type.is_list ? Java::IterableOf(input_type) : input_type);
    ParseDescription(input.description(), input_var.doc_ptr());
    tmpl.AddInput(input_var);
  }
  for (const auto& attr : op.attr()) {
    ResolvedType type = type_resolver.TypeOf(attr);
    if (!type.is_inferred) {
      const string attr_name = SnakeToCamelCase(attr.name());
      if (Java::IsGeneric(type.dt) && !type.is_list) {
        JavaVar attr_var = Java::Var(attr.name(),
            Java::Class("Class").param(type.dt));
        ParseDescription(attr.description(), attr_var.doc_ptr());
        tmpl.AddTypeAttribute(attr_var);
      } else {
        JavaVar attr_var = Java::Var(attr_name,
          type.is_list ? Java::ListOf(type.dt) : type.dt);
        ParseDescription(attr.description(), attr_var.doc_ptr());
        tmpl.AddAttribute(attr_var, attr.has_default_value());
      }
    }
  }
  for (const auto& output : op.output_arg()) {
    const string output_name = SnakeToCamelCase(output.name());
    const ResolvedType type = type_resolver.TypeOf(output, false);
    JavaType output_type = Java::Class("Output", "org.tensorflow")
        .param(type.dt);
    JavaVar output_var = Java::Var(output_name,
        type.is_list ? Java::ListOf(output_type) : output_type);
    ParseDescription(output.description(), output_var.doc_ptr());
    tmpl.AddOutput(output_var);
    if (Java::IsGeneric(type.dt) && !IsParamOf(type.dt, op_class)) {
      op_class.param(type.dt);
    }
  }
  tmpl.OpClass(op_class);
  tmpl.RenderToFile(output_dir, env);

  return Status::OK();
}


}  // namespace java
}  // namespace tensorflow
