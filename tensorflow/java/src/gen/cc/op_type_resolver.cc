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

JavaType OpTypeResolver::AttrType(const OpDef_AttrDef& attr, bool inferred) {
  string attr_type = attr.type();
  bool is_list = false;
  if (attr.type().compare(0, 5, "list(") == 0) {
    attr_type = attr_type.substr(5, attr.type().find_last_of(')') - 5);
    is_list = !inferred;  // only return list if explicit attributes
  }
  // <type> can be:
  //   "string", "int", "float", "bool", "type", "shape", or "tensor"
  //   "numbertype", "realnumbertype", "quantizedtype"
  //       (meaning "type" with a restriction on valid values)
  //   "{int32,int64}" or {realnumbertype,quantizedtype,string}"
  //       (meaning "type" with a restriction containing unions of value types)
  //   "{\"foo\", \"bar\n baz\"}", or "{'foo', 'bar\n baz'}"
  //       (meaning "string" with a restriction on valid values)
  //   "list(string)", ..., "list(tensor)", "list(numbertype)", ...
  //       (meaning lists of the above types)
  //   "int >= 2" (meaning "int" with a restriction on valid values)
  //   "list(string) >= 2", "list(int) >= 2"
  //       (meaning "list(string)" / "list(int)" with length at least 2)
  // <default>, if included, should use the Proto text format
  // of <type>.  For lists use [a, b, c] format.
  JavaType type;
  if (attr_type == "string") {
    type = Java::Class("String");
  } else if (attr_type == "int") {
    type = Java::Class("Integer");
  } else if (attr_type == "float") {
    type = Java::Class("Float");
  } else if (attr_type == "bool") {
    type = Java::Class("Boolean");
  } else if (attr_type == "shape") {
    type = Java::Class("Shape", "org.tensorflow");
  } else if (attr_type == "tensor") {
    type = Java::Class("Tensor", "org.tensorflow").param(Java::Wildcard());
  } else if (attr_type == "type") {
    if (inferred) {
      type = Java::Generic(string(1, next_generic_));
      next_generic_ = (next_generic_ == 'Z') ? 'A' : next_generic_ + 1;
    } else {
      type = Java::Enum("DataType", "org.tensorflow");
    }
  } else {
    LOG(WARNING) << "Unsupported attribute type \"" << attr_type << "\"";
    type = inferred ? Java::Wildcard() : Java::Class("Object");
  }
  return is_list ? Java::ListOf(type) : type;
}

ResolvedType OpTypeResolver::TypeOf(const OpDef_ArgDef& arg, JavaType base_type,
    const OpDef& op) {
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
      // FIXME since the list of tensors could be of different type, return a wildcard all the time?
    }
    if (!attr_name.empty()) {
      std::map<string, JavaType>::const_iterator it;
      it = inferred_attrs_.find(attr_name);
      if (it != inferred_attrs_.cend()) {
        result.tensor = it->second;
      } else {
        for (const auto& attr : op.attr()) {
          if (attr.name() == attr_name) {
            result.tensor = AttrType(attr, true);
            inferred_attrs_.insert(std::pair<string, JavaType>(attr_name, result.tensor));
            if (Java::IsGeneric(result.tensor)) {
              result.is_new_generic = true;
            }
            break;
          }
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
