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
      *writer << it->second << "[] " << array_name << " = new " << it->second
          << "[" << var_name << ".size()];" << endl
          << "for (int i = 0; i < " << array_name << ".length; ++i)"
          << beginb
          << array_name << "[i] = " << var_name << ".get(i);" << endl
          << endb
          << "opBuilder.setAttr(\"" << attr.name() << "\", "
          << array_name << ");" << endl;
    } else {
      *writer << "opBuilder.setAttr(\"" << attr.name() << "\", " << var_name
          << ".toArray(new " << type << "[" << var_name << ".size()]));"
          << endl;
    }
  } else {
    JavaType type = attr.type();
    *writer << "opBuilder.setAttr(\"" << attr.name() << "\", ";
    if (type == Java::Class("Class")) {
      *writer << "DataType.fromClass(" << attr.name() << "));" << endl;
    } else {
      *writer << var_name << ");" << endl;
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
  writer << JavaSnippet(io::JoinPath(kGenResourcePath, "licence.snippet.java"));
  JavaClassWriter* op_writer =
      writer.BeginClass(op_class, imports_, PUBLIC|FINAL);
  if (!opt_attrs_.empty()) {
    RenderOptionsClass(op_writer);
  }
  RenderFactoryMethod(op_writer);
  RenderMethods(op_writer, mode, single_type);
  op_writer->WriteFields(outputs_, PRIVATE);
  RenderConstructor(op_writer);
  op_writer->EndClass();
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

      JavaMethodWriter* set_writer = opt_writer->BeginMethod(setter, PUBLIC);
      *set_writer << "this." << var->name() << " = " << var->name()
          << ";" << endl
          << "return this;" << endl;
      set_writer->EndMethod();
    }
    opt_writer->WriteFields(opt_attrs_, PRIVATE);

    JavaMethod opt_ctor = Java::ConstructorFor(opt_class);
    opt_writer->BeginMethod(opt_ctor, PRIVATE)->EndMethod();
    opt_writer->EndClass();
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
  JavaMethodWriter* fct_writer = op_writer->BeginMethod(factory, PUBLIC|STATIC);
  *fct_writer << "OperationBuilder opBuilder = scope.graph().opBuilder(\""
      << op_name_ << "\", scope.makeOpName(\"" << op_name_ << "\"));"
      << endl;
  std::vector<JavaVar>::const_iterator var;
  for (var = inputs_.begin(); var != inputs_.end(); ++var) {
    if (Java::IsCollection(var->type())) {
      *fct_writer << "opBuilder.addInputList(Operands.asOutputs("
          << var->name() << "));" << endl;
    } else {
      *fct_writer << "opBuilder.addInput(" << var->name()
          << ".asOutput());" << endl;
    }
  }
  for (var = attrs_.begin(); var != attrs_.end(); ++var) {
    WriteSetAttrDirective(*var, fct_writer, false);
  }
  if (!opt_attrs_.empty()) {
    *fct_writer << "if (options != null)" << beginb
        << "for (Options opts : options)" << beginb;
    for (var = opt_attrs_.begin(); var != opt_attrs_.end(); ++var) {
      *fct_writer << "if (opts." << var->name() << " != null)" << beginb;
      WriteSetAttrDirective(*var, fct_writer, true);
      *fct_writer << endb;
    }
    *fct_writer << endb << endb;
  }
  *fct_writer << "return new " << op_class_ << "(opBuilder.build());" << endl;
  fct_writer->EndMethod();
}

void OpTemplate::RenderMethods(JavaClassWriter* op_writer, RenderMode mode,
    const JavaType& single_output_type) {
  std::vector<JavaVar>::const_iterator var;

  // Options setters
  for (var = opt_attrs_.begin(); var != opt_attrs_.end(); ++var) {
    JavaMethod setter = Java::Method(var->name(), Java::Class("Options"));
    setter.arg(*var);
    JavaMethodWriter* set_writer = op_writer->BeginMethod(setter, PUBLIC|STATIC);
    *set_writer << "return new Options()." << var->name() << "("
            << var->name() << ");" << endl;
    set_writer->EndMethod();
  }
  // Output getters
  for (var = outputs_.begin(); var != outputs_.end(); ++var) {
    JavaMethod getter = Java::Method(var->name(), var->type());
    getter.doc(var->doc());
    JavaMethodWriter* get_writer = op_writer->BeginMethod(getter, PUBLIC);
    *get_writer << "return " << var->name() << ";" << endl;
    get_writer->EndMethod();
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
    JavaMethodWriter* out_writer = op_writer->BeginMethod(as_output, PUBLIC);
    *out_writer << "return ";
    if (cast) {
      *out_writer << "(" << return_type << ") ";
    }
    *out_writer << output.name() << ";" << endl;
    out_writer->EndMethod();

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
    JavaMethodWriter* it_writer = op_writer->BeginMethod(iterator, PUBLIC);
    *it_writer << "return (" << return_type.name() + ") "
            << outputs_.front().name() << ".iterator();" << endl;
    it_writer->EndMethod();
  }
}

void OpTemplate::RenderConstructor(JavaClassWriter* op_writer) {
  JavaVar operation = Java::Var("operation",
      Java::Class("Operation", "org.tensorflow"));

  JavaMethod constructor = Java::ConstructorFor(op_class_).arg(operation);
  constructor.annotation(
      Java::Annot("SuppressWarnings").attrs("\"unchecked\""));  // FIXME not always required!

  JavaMethodWriter* ctr_writer = op_writer->BeginMethod(constructor, PRIVATE);
  *ctr_writer << "super(operation);" << endl
      << "int outputIdx = 0;" << endl;

  std::vector<JavaVar>::const_iterator var;
  for (var = outputs_.begin(); var != outputs_.end(); ++var) {
    if (Java::IsCollection(var->type())) {
      string var_length_name = var->name() + "Length";
      *ctr_writer << "int " << var_length_name
          << " = operation.outputListLength(\"" << var->name() << "\");"
          << endl
          << var->name() << " = Arrays.asList(";
      const JavaType& tensor_type = FindOutputTensorType(var->type());
      if (!Java::IsWildcard(tensor_type)) {
        *ctr_writer << "(" << var->type().params().front() << "[])";
      }
      *ctr_writer << "operation.outputList(outputIdx, " << var_length_name
          << "));" << endl
          << "outputIdx += " << var_length_name << ";" << endl;
    } else {
      *ctr_writer << var->name() << " = operation.output(outputIdx++);" << endl;
    }
  }
  ctr_writer->EndMethod();
}

}  // namespace java
}  // namespace tensorflow
