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
#include <vector>

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"
#include "tensorflow/java/src/gen/cc/java_writer.h"
#include "tensorflow/java/src/gen/cc/op_template.h"

namespace tensorflow {
namespace java {
namespace {

inline const JavaType& FindOutputTensorType(const JavaType& output_type) {
  if (Java::IsCollection(output_type)) {
    return output_type.params().front().params().front();
  }
  return output_type.params().front();
}

const std::map<string, JavaType> kPrimitiveAttrTypes = {
    { "Boolean", Java::Type("boolean") },
    { "Byte", Java::Type("byte") },
    { "Character", Java::Type("byte") },
    { "Float", Java::Type("float") },
    { "Integer", Java::Type("long") },
    { "Long", Java::Type("long") },
    { "Short", Java::Type("long") },
    { "Double", Java::Type("float") },
};

void WriteSetAttrDirective(const JavaVar& attr, JavaMethodWriter* writer,
    bool optional) {
  string var_name = optional ? "opts." + attr.name() : attr.name();
  if (Java::IsCollection(attr.type())) {
    const JavaType& type = attr.type().params().front();
    std::map<string, JavaType>::const_iterator it =
      kPrimitiveAttrTypes.find(type.name());
    if (it != kPrimitiveAttrTypes.end()) {
      string array_name = attr.name() + "Array";
      writer->Write(it->second)
          ->Write("[] " + array_name + " = new ")
          ->Write(it->second)
          ->WriteLine("[" + var_name + ".size()];");
      writer->BeginBlock("for (int i = 0; i < " + array_name + ".length; ++i)")
          ->WriteLine(array_name + "[i] = " + var_name + ".get(i);")
          ->EndOfBlock();
      writer->WriteLine("opBuilder.setAttr(\"" + attr.name() + "\", "
          + array_name + ");");
    } else {
      writer->Write("opBuilder.setAttr(\"" + attr.name() + "\", ")
          ->Write(var_name + ".toArray(new ")
          ->Write(type)
          ->WriteLine("[" + var_name + ".size()]));");
    }
  } else {
    JavaType type = attr.type();
    writer->Write("opBuilder.setAttr(\"" + attr.name() + "\", ");
    if (type == Java::Class("Class")) {
      writer->WriteLine("DataType.fromClass(" + attr.name() + "));");
    } else {
      writer->WriteLine(var_name + ");");
    }
  }
}

}  // namespace

OpTemplate::OpTemplate(const string& op_name) : op_name_(op_name) {
  imports_.insert({
    Java::Class("Operation", "org.tensorflow"),
    Java::Class("OperationBuilder", "org.tensorflow"),
    Java::Class("Scope", "org.tensorflow.op"),
  });
}

void OpTemplate::AddInput(const JavaVar& input) {
  AddVariable(input, &inputs_);
  if (Java::IsCollection(input.type())) {
    imports_.insert(Java::Class("Operands", "org.tensorflow.op"));
  }
}

void OpTemplate::AddOutput(const JavaVar& output) {
  AddVariable(output, &outputs_);
  if (Java::IsCollection(output.type())) {
    imports_.insert(Java::Class("Arrays", "java.util"));
  }
}

void OpTemplate::CollectImports(const JavaType& type) {
  auto import_scanner = [this](const JavaType* type) {
    if (!type->package().empty()) {
      this->imports_.insert(*type);
    }
  };
  type.Scan(&import_scanner);
}

void OpTemplate::RenderToFile(const string& root_dir, Env* env) {
  string package_path;
  if (!op_class_.package().empty()) {
    package_path = io::JoinPath(root_dir,
        str_util::StringReplace(op_class_.package(), ".", "/", true));
    if (!env->FileExists(package_path).ok()) {
      TF_CHECK_OK(env->RecursivelyCreateDir(package_path));
    }
  }
  string file_path = io::JoinPath(package_path, op_class_.name() + ".java");
  std::unique_ptr<tensorflow::WritableFile> file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(file_path, &file));
  SourceFileWriter src_writer(file.get());
  Render(&src_writer);
}

void OpTemplate::RenderToBuffer(string* buffer) {
  SourceBufferWriter src_writer(buffer);
  Render(&src_writer);
}

void OpTemplate::Render(SourceWriter* src_writer) {
  // Complete the effective op class to render by selecting supertypes
  JavaType op_class(op_class_);
  op_class.supertype(Java::Class("PrimitiveOp", "org.tensorflow.op"));
  JavaType single_type;

  RenderMode mode = DEFAULT;
  if (outputs_.size() == 1) {
      const JavaVar& output = outputs_.front();
      single_type = FindOutputTensorType(output.type());
      if (Java::IsWildcard(single_type)) {
        single_type = Java::Class("Object");
      }
      JavaType operand = Java::Interface("Operand", "org.tensorflow");
      operand.param(single_type);

      if (Java::IsCollection(output.type())) {
        mode = SINGLE_LIST_OUTPUT;
        op_class.supertype(Java::IterableOf(operand));
        imports_.insert(Java::Interface("Iterator", "java.util"));

      } else {
        mode = SINGLE_OUTPUT;
        op_class.supertype(operand);
        imports_.insert(Java::Class("Output", "org.tensorflow"));
      }
  }
  CollectImports(op_class);

  // Render the op class to the selected target
  JavaWriter writer(src_writer);
  writer.WriteSnippet(
      io::JoinPath(kGenResourcePath, "licence.snippet.java"));
  JavaClassWriter* op_writer =
      writer.BeginClass(op_class, imports_, PUBLIC|FINAL);
  if (!opt_attrs_.empty()) {
    RenderOptionsClass(op_writer);
  }
  RenderFactoryMethod(op_writer);
  RenderMethods(op_writer, mode, single_type);
  op_writer->WriteFields(outputs_, PRIVATE);
  RenderConstructor(op_writer);
  op_writer->EndOfClass();
}

void OpTemplate::RenderOptionsClass(JavaClassWriter* op_writer) {
    JavaType opt_class = Java::Class("Options");
    opt_class.doc_ptr()
        ->descr("Class holding optional attributes of this operation");

    JavaClassWriter* opt_writer =
        op_writer->BeginInnerClass(opt_class, PUBLIC|STATIC);

    std::vector<JavaVar>::const_iterator var;
    for (var = opt_attrs_.begin(); var != opt_attrs_.end(); ++var) {
      JavaMethod setter = Java::Method(var->name(), opt_class);
      setter.arg(*var);

      opt_writer->BeginMethod(setter, PUBLIC)
          ->WriteLine("this." + var->name() + " = " + var->name() + ";")
          ->WriteLine("return this;")
          ->EndOfMethod();
    }
    opt_writer->WriteFields(opt_attrs_, PRIVATE);

    JavaMethod opt_ctor = Java::ConstructorFor(opt_class);
    opt_writer->BeginMethod(opt_ctor, PRIVATE)->EndOfMethod();
    opt_writer->EndOfClass();
}

void OpTemplate::RenderFactoryMethod(JavaClassWriter* op_writer) {
  JavaVar scope = Java::Var("scope", Java::Class("Scope", "org.tensorflow.op"));
  scope.doc_ptr()->descr("Current graph scope");

  JavaMethod factory = Java::Method("create", op_class_);
  factory.doc_ptr()->descr("Factory method to create a class to wrap a new "
          + op_name_ + " operation to the graph.");
  factory.doc_ptr()->value("a new instance of " + op_class_.name());
  factory.arg(scope);
  factory.args(inputs_);
  factory.args(attrs_);
  if (!opt_attrs_.empty()) {
    JavaVar options = Java::PeriodicVar("options", Java::Class("Options"));
    options.doc_ptr()->descr("an object holding optional attributes values");
    factory.arg(options);
  }
  JavaMethodWriter* factory_writer =
      op_writer->BeginMethod(factory, PUBLIC|STATIC)
               ->WriteLine(string("OperationBuilder opBuilder = ")
                   + "scope.graph().opBuilder(\"" + op_name_
                   + "\", scope.makeOpName(\"" + op_name_ + "\"));");

  std::vector<JavaVar>::const_iterator var;
  for (var = inputs_.begin(); var != inputs_.end(); ++var) {
    if (Java::IsCollection(var->type())) {
      factory_writer->WriteLine(
          "opBuilder.addInputList(Operands.asOutputs(" + var->name() + "));");
    } else {
      factory_writer->WriteLine(
          "opBuilder.addInput(" + var->name() + ".asOutput());");
    }
  }
  for (var = attrs_.begin(); var != attrs_.end(); ++var) {
    WriteSetAttrDirective(*var, factory_writer, false);
  }
  if (!opt_attrs_.empty()) {
    factory_writer->BeginBlock("if (options != null)")
        ->BeginBlock("for (Options opts : options)");
    for (var = opt_attrs_.begin(); var != opt_attrs_.end(); ++var) {
      factory_writer->BeginBlock("if (opts." + var->name() + " != null)");
      WriteSetAttrDirective(*var, factory_writer, true);
      factory_writer->EndOfBlock();
    }
    factory_writer->EndOfBlock()->EndOfBlock();
  }
  factory_writer->Write("return new ")
      ->Write(op_class_)
      ->WriteLine("(opBuilder.build());")
      ->EndOfMethod();
}

void OpTemplate::RenderMethods(JavaClassWriter* op_writer, RenderMode mode,
    const JavaType& single_output_type) {
  std::vector<JavaVar>::const_iterator var;

  // Options setters
  for (var = opt_attrs_.begin(); var != opt_attrs_.end(); ++var) {
    JavaMethod setter = Java::Method(var->name(), Java::Class("Options"));
    setter.arg(*var);
    op_writer->BeginMethod(setter, PUBLIC|STATIC)
        ->WriteLine("return new Options()." + var->name() + "("
            + var->name() + ");")
        ->EndOfMethod();
  }
  // Output getters
  for (var = outputs_.begin(); var != outputs_.end(); ++var) {
    JavaMethod getter = Java::Method(var->name(), var->type());
    getter.doc(var->doc());
    op_writer->BeginMethod(getter, PUBLIC)
        ->WriteLine("return " + var->name() + ";")
        ->EndOfMethod();
  }
  // Interface methods
  if (mode == SINGLE_OUTPUT) {
    JavaType return_type = Java::Class("Output", "org.tensorflow")
        .param(single_output_type);
    JavaMethod as_output = Java::Method("asOutput", return_type)
        .annotation(Java::Annot("Override"));
    // cast the output if not of the same tensor type
    JavaVar output = outputs_.front();
    bool cast = single_output_type != FindOutputTensorType(output.type());
    if (cast) {
      as_output.annotation(
          Java::Annot("SuppressWarnings").attrs("\"unchecked\""));
    }
    JavaMethodWriter* method_writer =
        op_writer->BeginMethod(as_output, PUBLIC)->Write("return ");
    if (cast) {
      method_writer->Write("(")->Write(return_type)->Write(") ");
    }
    method_writer->WriteLine(output.name() + ";")->EndOfMethod();

  } else if (mode == SINGLE_LIST_OUTPUT) {
    JavaType operand = Java::Interface("Operand", "org.tensorflow");
    operand.param(single_output_type);
    JavaType return_type = Java::Interface("Iterator", "java.util");
    return_type.param(operand);
    JavaMethod iterator = Java::Method("iterator", return_type)
        .annotation(Java::Annot("Override"))
        .annotation(Java::Annot("SuppressWarnings")
            .attrs("{\"rawtypes\", \"unchecked\"}"));

    // cast the output list using a raw List
    op_writer->BeginMethod(iterator, PUBLIC)
        ->WriteLine("return (" + return_type.name() + ") "
            + outputs_.front().name() + ".iterator();")
        ->EndOfMethod();
  }
}

void OpTemplate::RenderConstructor(JavaClassWriter* op_writer) {
  JavaVar operation = Java::Var("operation",
      Java::Class("Operation", "org.tensorflow"));

  JavaMethod ctor = Java::ConstructorFor(op_class_).arg(operation);
  ctor.annotation(Java::Annot("SuppressWarnings").attrs("\"unchecked\""));  // FIXME not always required!

  JavaMethodWriter* ctor_writer = op_writer->BeginMethod(ctor, PRIVATE);
  ctor_writer->WriteLine("super(operation);")
      ->WriteLine("int outputIdx = 0;");

  std::vector<JavaVar>::const_iterator var;
  for (var = outputs_.begin(); var != outputs_.end(); ++var) {
    if (Java::IsCollection(var->type())) {
      string var_length_name = var->name() + "Length";
      ctor_writer->WriteLine("int " + var_length_name
          + " = operation.outputListLength(\"" + var->name() + "\");");

      ctor_writer->Write(var->name() + " = Arrays.asList(");
      const JavaType& tensor_type = FindOutputTensorType(var->type());
      if (!Java::IsWildcard(tensor_type)) {
        ctor_writer->Write("(")
            ->Write(var->type().params().front())
            ->Write("[])");
      }
      ctor_writer->Write("operation.outputList(outputIdx, ")
          ->WriteLine(var_length_name + "));")
          ->WriteLine("outputIdx += " + var_length_name + ";");

    } else {
      ctor_writer->WriteLine(var->name() + " = operation.output(outputIdx++);");
    }
  }
  ctor_writer->EndOfMethod();
}

}  // namespace java
}  // namespace tensorflow
