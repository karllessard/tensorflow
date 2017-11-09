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

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/java/src/gen/cc/op_type_resolver.h"

namespace tensorflow {
namespace java {

ResolvedType OpTypeResolver::TypeOf(const OpDef_AttrDef& attr,
    bool allow_generic) {
  std::map<string, ResolvedType>::const_iterator it;
  it = resolved_attrs_.find(attr.name());
  if (it != resolved_attrs_.cend()) {
    return it->second;
  }
  ResolvedType type;
  string attr_type = attr.type();
  if (attr.type().compare(0, 5, "list(") == 0) {
    attr_type = attr_type.substr(5, attr.type().find_last_of(')') - 5);
    type.is_list = true;
  }
  if (attr_type == "string") {
    type.dt = Java::Class("String");
  } else if (attr_type == "int") {
    type.dt = Java::Class("Integer");
  } else if (attr_type == "float") {
    type.dt = Java::Class("Float");
  } else if (attr_type == "bool") {
    type.dt = Java::Class("Boolean");
  } else if (attr_type == "shape") {
    type.dt = Java::Class("Shape", "org.tensorflow");
  } else if (attr_type == "tensor") {
    type.dt = Java::Class("Tensor", "org.tensorflow").param(Java::Wildcard());
  } else if (attr_type == "type") {
    if (!type.is_list && allow_generic) {
      type.dt = Java::Generic(string(1, next_generic_));
      next_generic_ = (next_generic_ == 'Z') ? 'A' : next_generic_ + 1;
      type.is_generic = true;
    } else {
      type.dt = Java::Enum("DataType", "org.tensorflow");
    }
  } else {
    LOG(WARNING) << "Unsupported attribute type \"" << attr_type << "\"";
    type.dt = type.is_list ? Java::Wildcard() : Java::Class("Object");
  }
  std::pair<string, ResolvedType> attr_pair(attr.name(), type);
  attr_pair.second.is_inferred = type.is_generic;
  resolved_attrs_.insert(attr_pair);
  return type;
}

ResolvedType OpTypeResolver::TypeOf(const OpDef_ArgDef& arg, const OpDef& op) {
  ResolvedType type;
  type.is_list = !arg.number_attr().empty();
  if (arg.type() != DataType::DT_INVALID) {
    switch (arg.type()) {
      case DataType::DT_BOOL:
        type.dt = Java::Class("Boolean");
        break;
      case DataType::DT_STRING:
        type.dt = Java::Class("String");
        break;
      case DataType::DT_FLOAT:
        type.dt = Java::Class("Float");
        break;
      case DataType::DT_DOUBLE:
        type.dt = Java::Class("Double");
        break;
      case DataType::DT_UINT8:
        type.dt = Java::Class("UInt8", "org.tensorflow.types");
        break;
      case DataType::DT_INT32:
        type.dt = Java::Class("Integer");
        break;
      case DataType::DT_INT64:
        type.dt = Java::Class("Long");
        break;
      default:
        LOG(WARNING) << "Unsupported data type " << arg.type() << " for arg \""
            + arg.name() + "\"";
        type.dt = Java::Wildcard();
        break;
    }
  } else {
    string attr_name = arg.type_attr();
    if (attr_name.empty()) {
      attr_name = arg.type_list_attr();
    }
    if (!attr_name.empty()) {
      for (const auto& attr : op.attr()) {
        if (attr.name() == attr_name) {
          type = TypeOf(attr, true);
          if (type.dt == Java::Enum("DataType", "org.tensorflow")) {
            type.dt = Java::Wildcard();
          }
          break;
        }
      }
    }
    if (type.dt.empty()) {
        LOG(ERROR) << "Can't resolve type attribute for arg \"" + arg.name() + "\"";
        type.dt = Java::Wildcard();
    }
  }
  return type;
}

}  // namespace java
}  // namespace tensorflow
