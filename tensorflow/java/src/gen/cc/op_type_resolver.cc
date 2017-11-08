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

JavaType OpTypeResolver::AttrType(const OpDef& op, const string& attr_name) {
  JavaType tensor_type = Java::Wildcard();
  for (const auto& attr : op.attr()) {
    if (attr.name() == attr_name) {
      string attr_type = attr.type();
      if (attr.type().compare(0, 5, "list(") == 0) {
        attr_type = attr_type.substr(5, attr.type().find_last_of(')') - 5);
      }
      if (attr_type == "string") {
        tensor_type = Java::Class("String");
      } else if (attr_type == "int") {
        tensor_type = Java::Class("Integer");
      } else if (attr_type == "float") {
        tensor_type = Java::Class("Float");
      } else if (attr_type == "bool") {
        tensor_type = Java::Class("Boolean");
      } else if (attr_type == "type") {
        tensor_type = Java::Generic(string(1, next_generic_name++));
        if (next_generic_name == 'Z') {
          next_generic_name = 'A';
        } else {
          ++next_generic_name;
        }
      } else {
        LOG(WARNING) << "Unsupported attribute type \"" << attr_type << "\"";
      }
      break;
    }
  }
  return tensor_type;
}

ResolvedType OpTypeResolver::TypeOf(const OpDef& op, const OpDef_ArgDef& arg,
    JavaType base_type) {
  ResolvedType result;
  bool is_list = !arg.number_attr().empty();
  if (arg.type() != DataType::DT_INVALID) {
    switch (arg.type()) {
      case DataType::DT_BOOL:
        result.tensor = Java::Class("Boolean");
        break;
      case DataType::DT_STRING:
        result.tensor = Java::Class("String");
        break;
      case DataType::DT_FLOAT:
        result.tensor = Java::Class("Float");
        break;
      case DataType::DT_DOUBLE:
        result.tensor = Java::Class("Double");
        break;
      case DataType::DT_UINT8:
        result.tensor = Java::Class("UInt8", "org.tensorflow.types");
        break;
      case DataType::DT_INT32:
        result.tensor = Java::Class("Integer");
        break;
      case DataType::DT_INT64:
        result.tensor = Java::Class("Long");
        break;
      default:
        LOG(WARNING) << "Unsupported data type " << arg.type()
            << " for arg \"" + arg.name() + "\"";
        break;
    }
  } else {
    string attr_name = arg.type_attr();
    if (attr_name.empty()) {
      attr_name = arg.type_list_attr();
      is_list = true;
    }
    if (!attr_name.empty()) {
      std::map<string, JavaType>::const_iterator it;
      it = known_type_attrs.find(attr_name);
      if (it != known_type_attrs.cend()) {
        result.tensor = it->second;
      } else {
        result.tensor = AttrType(op, attr_name);
        known_type_attrs.insert(std::pair<string, JavaType>(attr_name, result.tensor));
        if (Java::IsGeneric(result.tensor)) {
          result.is_new_generic = true;
        }
      }
    } else {
        LOG(ERROR) << "Can't resolve type attribute for arg \"" + arg.name() + "\"";
    }
  }
  if (result.tensor.empty()) {
    LOG(WARNING) << "Unsupported tensor type for arg \"" + arg.name() + "\"";
    result.tensor = Java::Wildcard();
  }
  JavaType type = base_type;
  type.param(result.tensor);
  result.var = is_list ? Java::ListOf(type) : type;
  return result;
}

}  // namespace java
}  // namespace tensorflow
