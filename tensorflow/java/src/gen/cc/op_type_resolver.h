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

#ifndef TENSORFLOW_JAVA_SRC_GEN_CC_OP_TYPE_RESOLVER_H_
#define TENSORFLOW_JAVA_SRC_GEN_CC_OP_TYPE_RESOLVER_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"

namespace tensorflow {
namespace java {

class TypeInfo {
 public:
  TypeInfo(const Type& type, bool list = false, bool inferred = false)
    : type_(type), collection_(list), inferred_(inferred) {}
  const Type& type() const { return type_; }
  bool collection() const { return collection_; }
  bool inferred() const { return inferred_; }

 private:
  Type type_;
  bool collection_;
  bool inferred_;  // only true for attribute types
};

class OpTypeResolver {
 public:
  OpTypeResolver() {}
  virtual ~OpTypeResolver() {}

  TypeInfo TypeOf(const OpDef_ArgDef& arg, const OpDef& op, bool is_input);
  TypeInfo TypeOf(const OpDef_AttrDef& attr, bool is_inferred = false);

 private:
  std::map<string, TypeInfo> resolved_attrs_;
  char next_generic_ = 'T';

  Type AttrType(const OpDef_AttrDef& attr, bool list);
  Type ArgType(const DataType arg_type);
  Type NextGenericType(const AttrValue& allowed_values);
};

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_OP_TYPE_RESOLVER_H_
