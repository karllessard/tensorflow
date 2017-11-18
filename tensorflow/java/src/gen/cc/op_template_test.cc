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

JavaDoc GenerateDoc(const string& name) {
  JavaDoc doc;
  doc.brief("This is a short description of this " + name);
  doc.descr("This is a longer description of this " + name
     + " with multiple line breaks\n\n\nthat should all be prefixed by a star");
  return doc;
}

void RenderSingleOutputOp(const TestFunc& tester) {
  OpTemplate tmpl("Test");

  JavaType tensor_type = Java::Generic("T")
      .supertype(Java::Class("Number"));
  JavaType op_class = Java::Class("SingleOutputOp", "org.tensorflow.op.test")
      .doc(GenerateDoc("class"))
      .param(tensor_type)
      .annotation(Java::Annot("Operator", "org.tensorflow.op.annotation")
          .attrs("group = \"test\""));
  tmpl.OpClass(op_class);

  JavaType input_type = Java::Interface("Operand", "org.tensorflow")
      .param(tensor_type);
  JavaVar input = Java::Var("input", input_type)
      .doc(GenerateDoc("input"));
  tmpl.AddInput(input);

  JavaType output_type = Java::Class("Output", "org.tensorflow")
      .param(tensor_type);
  JavaVar output = Java::Var("output", output_type)
      .doc(GenerateDoc("output"));
  tmpl.AddOutput(output);

  tester(&tmpl, op_class.name());
}

void RenderSingleOutputListOp(const TestFunc& tester) {
  OpTemplate tmpl("Test");

  JavaType tensor_type = Java::Generic("T")
      .supertype(Java::Class("Number"));
  JavaType op_class = Java::Class("SingleOutputListOp", "org.tensorflow.op.test")
      .doc(GenerateDoc("class"))
      .param(tensor_type)
      .annotation(Java::Annot("Operator", "org.tensorflow.op.annotation")
          .attrs("group = \"test\""));
  tmpl.OpClass(op_class);

  JavaType input_type = Java::Interface("Operand", "org.tensorflow")
      .param(tensor_type);
  JavaVar input = Java::Var("input", input_type)
      .doc(GenerateDoc("input"));
  tmpl.AddInput(input);

  JavaType output_type = Java::Class("Output", "org.tensorflow")
      .param(tensor_type);
  JavaVar output = Java::Var("outputList", Java::ListOf(output_type))
      .doc(GenerateDoc("outputList"));
  tmpl.AddOutput(output);

  tester(&tmpl, op_class.name());
}

void RenderMultipleAndMixedOutputsOp(const TestFunc& tester) {
  OpTemplate tmpl("Test");

  JavaType tensor_a_type = Java::Generic("T");
  JavaType tensor_b_type = Java::Generic("U");
  JavaType op_class =
      Java::Class("MultipleAndMixedOutputsOp", "org.tensorflow.op.test")
          .doc(GenerateDoc("class"))
          .param(tensor_a_type)
          .param(tensor_b_type)
          .annotation(Java::Annot("Operator", "org.tensorflow.op.annotation")
              .attrs("group = \"test\""));
  tmpl.OpClass(op_class);

  JavaType input_a_type = Java::Interface("Operand", "org.tensorflow")
      .param(tensor_a_type);
  JavaVar input_a = Java::Var("input", input_a_type)
      .doc(GenerateDoc("input"));
  tmpl.AddInput(input_a);

  JavaType input_b_type = Java::Interface("Operand", "org.tensorflow")
      .param(tensor_a_type);
  JavaVar input_b = Java::Var("inputList", Java::IterableOf(input_b_type))
      .doc(GenerateDoc("inputList"));
  tmpl.AddInput(input_b);

  JavaType output_a_type = Java::Class("Output", "org.tensorflow")
      .param(tensor_a_type);
  JavaVar output_a = Java::Var("output", output_a_type)
      .doc(GenerateDoc("output"));
  tmpl.AddOutput(output_a);

  JavaType output_b_type = Java::Class("Output", "org.tensorflow")
      .param(tensor_b_type);
  JavaVar output_b = Java::Var("outputList", Java::ListOf(output_b_type))
      .doc(GenerateDoc("output list"));
  tmpl.AddOutput(output_b);

  tester(&tmpl, op_class.name());
}

void RenderOptionalAndMandatoryAttributesOp(const TestFunc& tester) {
  OpTemplate tmpl("Test");

  JavaType tensor_type = Java::Generic("T");
  JavaType op_class =
      Java::Class("OptionalAndMandatoryAttributesOp", "org.tensorflow.op.test")
          .doc(GenerateDoc("class"))
          .param(tensor_type)
          .annotation(Java::Annot("Operator", "org.tensorflow.op.annotation")
              .attrs("group = \"test\""));
  tmpl.OpClass(op_class);

  JavaType input_type = Java::Interface("Operand", "org.tensorflow")
      .param(tensor_type);
  JavaVar input = Java::Var("input", input_type)
      .doc(GenerateDoc("input"));
  tmpl.AddInput(input);

  JavaVar attr = Java::Var("attr", Java::Class("Integer"))
      .doc(GenerateDoc("attribute"));
  tmpl.AddAttribute(attr, false);

  JavaVar opt_attr = Java::Var("opt", Java::Class("Integer"))
      .doc(GenerateDoc("optional attribute"));
  tmpl.AddAttribute(opt_attr, true);

  JavaType output_type = Java::Class("Output", "org.tensorflow")
      .param(tensor_type);
  JavaVar output = Java::Var("output", output_type)
      .doc(GenerateDoc("output"));
  tmpl.AddOutput(output);

  tester(&tmpl, op_class.name());
}

void RenderTypedInputListOp(const TestFunc& tester) {
  OpTemplate tmpl("Test");

  JavaType tensor_type = Java::Class("Integer");
  JavaType op_class =
      Java::Class("TypedInputListOp", "org.tensorflow.op.test")
          .doc(GenerateDoc("class"))
          .annotation(Java::Annot("Operator", "org.tensorflow.op.annotation")
              .attrs("group = \"test\""));
  tmpl.OpClass(op_class);

  JavaType input_type = Java::Interface("Operand", "org.tensorflow")
      .param(tensor_type);
  JavaVar input_list = Java::Var("inputList", Java::IterableOf(input_type))
      .doc(GenerateDoc("inputList"));
  tmpl.AddInput(input_list);

  JavaType output_type = Java::Class("Output", "org.tensorflow")
      .param(tensor_type);
  JavaVar output = Java::Var("output", output_type)
      .doc(GenerateDoc("output"));
  tmpl.AddOutput(output);

  tester(&tmpl, op_class.name());
}

void RenderWildcardInputListOp(const TestFunc& tester) {
  OpTemplate tmpl("Test");

  JavaType op_class =
      Java::Class("WildcardInputListOp", "org.tensorflow.op.test")
          .doc(GenerateDoc("class"))
          .annotation(Java::Annot("Operator", "org.tensorflow.op.annotation")
              .attrs("group = \"test\""));
  tmpl.OpClass(op_class);

  JavaType input_type = Java::Interface("Operand", "org.tensorflow")
      .param(Java::Wildcard());
  JavaVar input_list = Java::Var("inputList", Java::IterableOf(input_type))
      .doc(GenerateDoc("inputList"));
  tmpl.AddInput(input_list);

  JavaType output_type = Java::Class("Output", "org.tensorflow")
      .param(Java::Wildcard());
  JavaVar output = Java::Var("output", output_type)
      .doc(GenerateDoc("output"));
  tmpl.AddOutput(output);

  tester(&tmpl, op_class.name());
}

void RenderDeclaredOutputTypeOp(const TestFunc& tester) {
  OpTemplate tmpl("Test");

  JavaType tensor_type = Java::Generic("T")
      .supertype(Java::Class("Number"));
  JavaType op_class =
      Java::Class("DeclaredOutputTypeOp", "org.tensorflow.op.test")
          .param(tensor_type)
          .doc(GenerateDoc("class"))
          .annotation(Java::Annot("Operator", "org.tensorflow.op.annotation")
              .attrs("group = \"test\""));
  tmpl.OpClass(op_class);

  JavaType input_type = Java::Interface("Operand", "org.tensorflow")
      .param(Java::Wildcard());
  JavaVar input = Java::Var("input", input_type)
      .doc(GenerateDoc("input"));
  tmpl.AddInput(input);

  JavaType output_type = Java::Class("Output", "org.tensorflow")
      .param(tensor_type);
  JavaVar output = Java::Var("output", output_type)
      .doc(GenerateDoc("output"));
  tmpl.AddOutput(output, true);

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
