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

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/java/src/gen/cc/op_type_resolver.h"

namespace tensorflow {
namespace java {
namespace {

bool IsRealNumber(DataType type) {
  for (DataType dt : RealNumberTypes()) {
    if (type == dt) {
      return true;
    }
  }
  return false;
}

bool IsRealNumbers(const AttrValue& values) {
  if (values.has_list()) {
    for (int i = 0; i < values.list().type_size(); ++i) {
      if (!IsRealNumber(values.list().type(i))) {
        return false;
      }
    }
    return true;
  }
  return IsRealNumber(values.type());
}

}  // namespace

TypeInfo OpTypeResolver::TypeOf(const OpDef_AttrDef& attr,
    bool inferred) {
  std::map<string, TypeInfo>::const_iterator attr_type_it =
      resolved_attrs_.find(attr.name());
  if (attr_type_it != resolved_attrs_.cend()) {
    return attr_type_it->second;
  }
  bool list = false;
  if (attr.type().compare(0, 5, "list(") == 0) {
    list = true;
  }
  Type type = AttrType(attr, list);
  std::pair<string, TypeInfo> attr_pair(attr.name(), type);
  resolved_attrs_.insert(attr_pair);
  return TypeInfo(type, list, inferred);
}

TypeInfo OpTypeResolver::TypeOf(const OpDef_ArgDef& arg, const OpDef& op,
    bool is_input) {
  bool list = false;
  if (!arg.number_attr().empty()) {
    // Resolve number attribute so we remember later that it could be inferred
    for (const auto& attr : op.attr()) {
      if (attr.name() == arg.number_attr()) {
        TypeOf(attr, true);
      }
    }
    list = true;
  }
  if (arg.type() == DataType::DT_INVALID) {
    string attr_name = arg.type_attr();
    if (attr_name.empty()) {
      attr_name = arg.type_list_attr();
      list = true;
    }
    for (const auto& attr : op.attr()) {
      if (attr.name() == attr_name) {
        Type attr_type = list ? Type::Wildcard() : AttrType(attr, list);
        return TypeInfo(attr_type, list);
      }
    }
  }
  return TypeInfo(ArgType(arg.type()), list);
}

Type OpTypeResolver::NextGenericType(const AttrValue& allowed_values)  {
  Type generic = Type::Generic(string(1, next_generic_));
  // if allowed types only include real numbers, enforce that the passed
  // datatype extends java.lang.Number
  if (IsRealNumbers(allowed_values)) {
    generic.supertype(Type::Class("Number"));
  }
  next_generic_ = (next_generic_ == 'Z') ? 'A' : next_generic_ + 1;
  return generic;
}

Type OpTypeResolver::AttrType(const OpDef_AttrDef& attr, bool list) {
  string attr_type = attr.type();
  if (list) {
    attr_type = attr_type.substr(5, attr.type().find_last_of(')') - 5);
    list = true;
  }
  if (attr_type == "type") {
    if (list) {
      return Type::Enum("DataType", "org.tensorflow");
    }
    return NextGenericType(attr.allowed_values());
  }
  if (attr_type == "string") {
    return Type::Class("String");
  }
  if (attr_type == "int") {
    return Type::Class("Integer");
  }
  if (attr_type == "float") {
    return Type::Class("Float");
  }
  if (attr_type == "bool") {
    return Type::Class("Boolean");
  }
  if (attr_type == "shape") {
    return Type::Class("Shape", "org.tensorflow");
  }
  if (attr_type == "tensor") {
    return Type::Class("Tensor", "org.tensorflow").param(Type::Wildcard());
  }
  LOG(WARNING) << "Unsupported attribute type \"" << attr_type << "\"";
  return list ? Type::Wildcard() : Type::Class("Object");
}

Type OpTypeResolver::ArgType(const DataType arg_type) {
  switch (arg_type) {
    case DataType::DT_BOOL:
      return Type::Class("Boolean");
    case DataType::DT_STRING:
      return Type::Class("String");
    case DataType::DT_FLOAT:
      return Type::Class("Float");
    case DataType::DT_DOUBLE:
      return Type::Class("Double");
    case DataType::DT_UINT8:
      return Type::Class("UInt8", "org.tensorflow.types");
    case DataType::DT_INT32:
      return Type::Class("Integer");
    case DataType::DT_INT64:
      return Type::Class("Long");
    case DataType::DT_RESOURCE:
      // TODO (karllessard) Create a Resource utility class that could be
      // used to store a resource and its type (passed in a second argument).
      // For now, we need to force a wildcard and we will unfortunately lose
      // track of the resource type.
      return Type::Wildcard();
    default:
      LOG(WARNING) << "Unsupported data type \"" << arg_type << "\"";
      break;
  }
  return Type::Wildcard();
}


}  // namespace java
}  // namespace tensorflow
