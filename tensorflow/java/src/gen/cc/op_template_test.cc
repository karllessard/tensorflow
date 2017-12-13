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

#include <memory>
#include <iostream>
#include <list>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/java/src/gen/cc/op_template.h"

namespace tensorflow {
namespace java {

typedef std::function<void(OpTemplate*, const string&)> TestFunc;

inline string GenerateDoc(const string& name) {
  return "This is a longer description of this " + name
     + " with multiple line breaks\n\n\nthat should all be prefixed by a star";
}

void RenderSingleOutputOp(const TestFunc& tester) {
  Type tensor_type = Type::Generic("T")
      .supertype(Type::Class("Number"));
  Type op_class = Type::Class("SingleOutputOp", "org.tensorflow.op.test")
      .descr(GenerateDoc("class"))
      .param(tensor_type)
      .annotation(Annotation::Of("Operator", "org.tensorflow.op.annotation")
          .attrs("group = \"test\""));

  OpTemplate tmpl("Test", op_class);

  Type input_type = Type::Interface("Operand", "org.tensorflow")
      .param(tensor_type);
  Variable input = Variable::Of("input", input_type)
      .descr(GenerateDoc("input"));
  tmpl.AddInput(input);

  Type output_type = Type::Class("Output", "org.tensorflow")
      .param(tensor_type);
  Variable output = Variable::Of("output", output_type)
      .descr(GenerateDoc("output"));
  tmpl.AddOutput(output);

  tester(&tmpl, op_class.name());
}

void RenderSingleOutputListOp(const TestFunc& tester) {
  Type tensor_type = Type::Generic("T")
      .supertype(Type::Class("Number"));
  Type op_class = Type::Class("SingleOutputListOp", "org.tensorflow.op.test")
      .descr(GenerateDoc("class"))
      .param(tensor_type)
      .annotation(Annotation::Of("Operator", "org.tensorflow.op.annotation")
          .attrs("group = \"test\""));

  OpTemplate tmpl("Test", op_class);

  Type input_type = Type::Interface("Operand", "org.tensorflow")
      .param(tensor_type);
  Variable input = Variable::Of("input", input_type)
      .descr(GenerateDoc("input"));
  tmpl.AddInput(input);

  Type output_type = Type::Class("Output", "org.tensorflow")
      .param(tensor_type);
  Variable output = Variable::Of("outputList", Type::ListOf(output_type))
      .descr(GenerateDoc("outputList"));
  tmpl.AddOutput(output);

  tester(&tmpl, op_class.name());
}

void RenderMultipleAndMixedOutputsOp(const TestFunc& tester) {
  Type tensor_a_type = Type::Generic("T");
  Type tensor_b_type = Type::Generic("U");
  Type op_class =
      Type::Class("MultipleAndMixedOutputsOp", "org.tensorflow.op.test")
          .descr(GenerateDoc("class"))
          .param(tensor_a_type)
          .param(tensor_b_type)
          .annotation(Annotation::Of("Operator", "org.tensorflow.op.annotation")
              .attrs("group = \"test\""));

  OpTemplate tmpl("Test", op_class);

  Type input_a_type = Type::Interface("Operand", "org.tensorflow")
      .param(tensor_a_type);
  Variable input_a = Variable::Of("input", input_a_type)
      .descr(GenerateDoc("input"));
  tmpl.AddInput(input_a);

  Type input_b_type = Type::Interface("Operand", "org.tensorflow")
      .param(tensor_a_type);
  Variable input_b = Variable::Of("inputList", Type::IterableOf(input_b_type))
      .descr(GenerateDoc("inputList"));
  tmpl.AddInput(input_b);

  Type output_a_type = Type::Class("Output", "org.tensorflow")
      .param(tensor_a_type);
  Variable output_a = Variable::Of("output", output_a_type)
      .descr(GenerateDoc("output"));
  tmpl.AddOutput(output_a);

  Type output_b_type = Type::Class("Output", "org.tensorflow")
      .param(tensor_b_type);
  Variable output_b = Variable::Of("outputList", Type::ListOf(output_b_type))
      .descr(GenerateDoc("output list"));
  tmpl.AddOutput(output_b);

  tester(&tmpl, op_class.name());
}

void RenderOptionalAndMandatoryAttributesOp(const TestFunc& tester) {
  Type tensor_type = Type::Generic("T");
  Type op_class =
      Type::Class("OptionalAndMandatoryAttributesOp", "org.tensorflow.op.test")
          .descr(GenerateDoc("class"))
          .param(tensor_type)
          .annotation(Annotation::Of("Operator", "org.tensorflow.op.annotation")
              .attrs("group = \"test\""));

  OpTemplate tmpl("Test", op_class);

  Type input_type = Type::Interface("Operand", "org.tensorflow")
      .param(tensor_type);
  Variable input = Variable::Of("input", input_type)
      .descr(GenerateDoc("input"));
  tmpl.AddInput(input);

  Variable attr = Variable::Of("attr", Type::Class("Integer"))
      .descr(GenerateDoc("attribute"));
  tmpl.AddAttribute(attr, false);

  Variable opt_attr = Variable::Of("opt", Type::Class("Integer"))
      .descr(GenerateDoc("optional attribute"));
  tmpl.AddAttribute(opt_attr, true);

  Type output_type = Type::Class("Output", "org.tensorflow")
      .param(tensor_type);
  Variable output = Variable::Of("output", output_type)
      .descr(GenerateDoc("output"));
  tmpl.AddOutput(output);

  tester(&tmpl, op_class.name());
}

void RenderTypedInputListOp(const TestFunc& tester) {
  Type tensor_type = Type::Class("Integer");
  Type op_class =
      Type::Class("TypedInputListOp", "org.tensorflow.op.test")
          .descr(GenerateDoc("class"))
          .annotation(Annotation::Of("Operator", "org.tensorflow.op.annotation")
              .attrs("group = \"test\""));

  OpTemplate tmpl("Test", op_class);

  Type input_type = Type::Interface("Operand", "org.tensorflow")
      .param(tensor_type);
  Variable input_list = Variable::Of("inputList", Type::IterableOf(input_type))
      .descr(GenerateDoc("inputList"));
  tmpl.AddInput(input_list);

  Type output_type = Type::Class("Output", "org.tensorflow")
      .param(tensor_type);
  Variable output = Variable::Of("output", output_type)
      .descr(GenerateDoc("output"));
  tmpl.AddOutput(output);

  tester(&tmpl, op_class.name());
}

void RenderWildcardInputListOp(const TestFunc& tester) {
  Type op_class =
      Type::Class("WildcardInputListOp", "org.tensorflow.op.test")
          .descr(GenerateDoc("class"))
          .annotation(Annotation::Of("Operator", "org.tensorflow.op.annotation")
              .attrs("group = \"test\""));

  OpTemplate tmpl("Test", op_class);

  Type input_type = Type::Interface("Operand", "org.tensorflow")
      .param(Type::Wildcard());
  Variable input_list = Variable::Of("inputList", Type::IterableOf(input_type))
      .descr(GenerateDoc("inputList"));
  tmpl.AddInput(input_list);

  Type output_type = Type::Class("Output", "org.tensorflow")
      .param(Type::Wildcard());
  Variable output = Variable::Of("output", output_type)
      .descr(GenerateDoc("output"));
  tmpl.AddOutput(output);

  tester(&tmpl, op_class.name());
}

void RenderDeclaredOutputTypeOp(const TestFunc& tester) {
  Type tensor_type = Type::Generic("T")
      .supertype(Type::Class("Number"));
  Type op_class =
      Type::Class("DeclaredOutputTypeOp", "org.tensorflow.op.test")
          .param(tensor_type)
          .descr(GenerateDoc("class"))
          .annotation(Annotation::Of("Operator", "org.tensorflow.op.annotation")
              .attrs("group = \"test\""));

  OpTemplate tmpl("Test", op_class);

  Type input_type = Type::Interface("Operand", "org.tensorflow")
      .param(Type::Wildcard());
  Variable input = Variable::Of("input", input_type)
      .descr(GenerateDoc("input"));
  tmpl.AddInput(input);

  Type output_type = Type::Class("Output", "org.tensorflow")
      .param(tensor_type);
  Variable output = Variable::Of("output", output_type)
      .descr(GenerateDoc("output"));
  tmpl.AddOutput(output);

  tester(&tmpl, op_class.name());
}

}  // namespace java
}  // namespace tensorflow

int main(int argc, char** argv) {
  std::cout << "Running main() from test_main.cc\n";
  tensorflow::Env* env = tensorflow::Env::Default();
  std::string out_dir;
  tensorflow::java::TestFunc tester;

  if (argc > 1) {
    // Gen mode: First argument is a directory where to output generated
    // source files
    out_dir = tensorflow::io::JoinPath(argv[1], "tensorflow/java/ops/test");
    if (!env->FileExists(out_dir).ok()) {
      env->RecursivelyCreateDir(out_dir);
    }
    tester = [out_dir](tensorflow::java::OpTemplate* tmpl) {
        tmpl->RenderToFile(out_dir);
    };

  } else {
    // Test mode: Simply render source code in memory and test
    tester = [](tensorflow::java::OpTemplate* tmpl, const std::string& name) {
        std::string buffer;
        tmpl->RenderToBuffer(&buffer);
        // TODO add test checks
    };
  }
  tensorflow::java::RenderSingleOutputOp(tester);
  tensorflow::java::RenderSingleOutputListOp(tester);
  tensorflow::java::RenderMultipleAndMixedOutputsOp(tester);
  tensorflow::java::RenderOptionalAndMandatoryAttributesOp(tester);
  tensorflow::java::RenderTypedInputListOp(tester);
  tensorflow::java::RenderWildcardInputListOp(tester);
  tensorflow::java::RenderDeclaredOutputTypeOp(tester);
}
