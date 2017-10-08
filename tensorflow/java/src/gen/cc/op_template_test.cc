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

#include <iostream>
#include <list>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/java/src/gen/cc/op_template.h"
#include "tensorflow/java/src/gen/cc/src_ostream.h"

namespace tensorflow {
namespace op_gen {

JavaDoc GenerateDoc(const string& name) {
  JavaDoc doc;
  doc.brief("This is a short description of this " + name);
  doc.description("This is a longer description of this " + name
     + " with multiple line breaks\n\n\nthat should all be prefixed by a star");
  return doc;
}

void RenderMultipleOutputsOp(const string& output_dir) {
  OpTemplate tmpl("Test", "test");

  JavaClass op_class("MultipleOutputsOp", "org.tensorflow.op.test");
  op_class.doc(GenerateDoc("class"));
  tmpl.ClassTemplate(op_class);

  JavaVariable input("input", JavaType("Operand", "org.tensorflow"));
  input.doc(GenerateDoc("input"));
  tmpl.AddInput(input);

  JavaVariable output("output", JavaType("Output", "org.tensorflow"));
  output.doc(GenerateDoc("output"));
  tmpl.AddOutput(output);

  JavaVariable output_list("outputList", JavaType("List", "java.util"));
  output_list.type_ptr()->param(JavaType("Output", "org.tensorflow"));
  output_list.doc(GenerateDoc("output list"));
  tmpl.AddOutput(output_list);

  FileSourceOutputStream stream(
      io::JoinPath(output_dir, "MultipleOutputsOp.java"));
  tmpl.Render(&stream);
}

void RenderMultipleOutputsOpWithOptions(const string& output_dir) {
  OpTemplate tmpl("Test", "test");

  JavaClass op_class("MultipleOutputsOpWithOptions", "org.tensorflow.op.test");
  op_class.doc(GenerateDoc("class"));
  tmpl.ClassTemplate(op_class);

  JavaVariable input("input", JavaType("Operand", "org.tensorflow"));
  input.doc(GenerateDoc("input"));
  tmpl.AddInput(input);

  JavaVariable opt_attr("opt", JavaType("Integer"));
  opt_attr.doc(GenerateDoc("optional attribute"));
  tmpl.AddAttribute(opt_attr, true);

  JavaVariable output("output", JavaType("Output", "org.tensorflow"));
  output.doc(GenerateDoc("output"));
  tmpl.AddOutput(output);

  JavaVariable output_list("outputList", JavaType("List", "java.util"));
  output_list.type_ptr()->param(JavaType("Output", "org.tensorflow"));
  output_list.doc(GenerateDoc("output list"));
  tmpl.AddOutput(output_list);

  FileSourceOutputStream stream(
      io::JoinPath(output_dir, "MultipleOutputsOpWithOptions.java"));
  tmpl.Render(&stream);
}

void RenderSingleOutputOp(const string& output_dir) {
  OpTemplate tmpl("Test", "test");

  JavaClass op_class("SingleOutputOp", "org.tensorflow.op.test");
  op_class.doc(GenerateDoc("class"));
  tmpl.ClassTemplate(op_class);

  JavaVariable input("input", JavaType("Operand", "org.tensorflow"));
  input.doc(GenerateDoc("input"));
  tmpl.AddInput(input);

  JavaVariable output("output", JavaType("Output", "org.tensorflow"));
  output.doc(GenerateDoc("output"));
  tmpl.AddOutput(output);

  FileSourceOutputStream stream(
      io::JoinPath(output_dir, "SingleOutputOp.java"));
  tmpl.Render(&stream);
}

void RenderSingleOutputListOp(const string& output_dir) {
  OpTemplate tmpl("Test", "test");

  JavaClass op_class("SingleOutputListOp", "org.tensorflow.op.test");
  op_class.doc(GenerateDoc("class"));
  tmpl.ClassTemplate(op_class);

  JavaVariable input("input", JavaType("Operand", "org.tensorflow"));
  input.doc(GenerateDoc("input"));
  tmpl.AddInput(input);

  JavaVariable output_list("outputList", JavaType("List", "java.util"));
  output_list.type_ptr()->param(JavaType("Output", "org.tensorflow"));
  output_list.doc(GenerateDoc("output list"));
  tmpl.AddOutput(output_list);

  FileSourceOutputStream stream(
      io::JoinPath(output_dir, "SingleOutputListOp.java"));
  tmpl.Render(&stream);
}

void RenderGenericOp(const string& output_dir) {
  OpTemplate tmpl("Test", "test");

  JavaClass op_class("GenericOp", "org.tensorflow.op.test");
  op_class.doc(GenerateDoc("class"));
  JavaType tensor_type("T");
  tensor_type.generic(true);
  op_class.param(tensor_type);
  tmpl.ClassTemplate(op_class);

  JavaType input_type("Operand", "org.tensorflow");
  input_type.param(tensor_type);
  JavaVariable input("input", input_type);
  input.doc(GenerateDoc("input"));
  tmpl.AddInput(input);

  JavaType input_list_type("Iterable");
  input_list_type.param(input_type);
  JavaVariable input_list("inputList", input_list_type);
  input_list.doc(GenerateDoc("input list"));
  tmpl.AddInput(input_list);

  JavaVariable attr("attr", JavaType("Boolean"));
  attr.doc(GenerateDoc("attribute"));
  tmpl.AddAttribute(attr, false);

  JavaVariable opt_attr("opt", JavaType("Integer"));
  opt_attr.doc(GenerateDoc("optional attribute"));
  tmpl.AddAttribute(opt_attr, true);

  JavaType output_type("Output", "org.tensorflow");
  output_type.param(tensor_type);
  JavaVariable output("output", output_type);
  output.doc(GenerateDoc("output"));
  tmpl.AddOutput(output);

  JavaType output_list_type("List", "java.util");
  output_list_type.param(output_type);
  JavaVariable output_list("outputList", output_list_type);
  output_list.doc(GenerateDoc("output list"));
  tmpl.AddOutput(output_list);

  FileSourceOutputStream stream(io::JoinPath(output_dir, "GenericOp.java"));
  tmpl.Render(&stream);
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
  std::cout << "Running main() from test_main.cc\n";
  tensorflow::Env* env = tensorflow::Env::Default();
  std::string output_dir =
      tensorflow::io::JoinPath(argv[1], "tensorflow/java/ops/test/");
  if (!env->FileExists(output_dir).ok()) {
    env->RecursivelyCreateDir(output_dir);
  }
  tensorflow::op_gen::RenderMultipleOutputsOp(output_dir);
  tensorflow::op_gen::RenderMultipleOutputsOpWithOptions(output_dir);
  tensorflow::op_gen::RenderSingleOutputOp(output_dir);
  tensorflow::op_gen::RenderSingleOutputListOp(output_dir);
  tensorflow::op_gen::RenderGenericOp(output_dir);
}
