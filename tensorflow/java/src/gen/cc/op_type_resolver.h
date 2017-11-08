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

struct ResolvedType {
  JavaType var;
  JavaType tensor;
  bool is_new_generic = false;
};

class OpTypeResolver {
 public:
  OpTypeResolver() {}
  virtual ~OpTypeResolver() {}

  ResolvedType InputType(const OpDef_ArgDef& input, const OpDef& op) {
    return TypeOf(input, Java::Interface("Operand", "org.tensorflow"), op);
  }
  ResolvedType OutputType(const OpDef_ArgDef& output, const OpDef& op) {
    return TypeOf(output, Java::Class("Output", "org.tensorflow"), op);
  }
  JavaType AttrType(const OpDef_AttrDef& attr, bool inferred = false);
  bool IsInferred(const OpDef_AttrDef& attr) {
    return inferred_attrs_.find(attr.name()) != inferred_attrs_.end();
  }

 private:
  std::map<string, JavaType> inferred_attrs_;
  char next_generic_ = 'T';

  ResolvedType TypeOf(const OpDef_ArgDef& arg, JavaType base_type,
      const OpDef& op);
};

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_OP_TYPE_RESOLVER_H_
