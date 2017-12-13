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

inline const Type& FindOutputTensorType(const Type& output_type) {
  if (output_type.IsCollection()) {
    return output_type.params().front().params().front();
  }
  return output_type.params().front();
}

const std::map<string, Type> kPrimitiveAttrTypes = {
    { "Boolean", Type::Primitive("boolean") },
    { "Byte", Type::Primitive("byte") },
    { "Character", Type::Primitive("byte") },
    { "Float", Type::Primitive("float") },
    { "Integer", Type::Primitive("long") },
    { "Long", Type::Primitive("long") },
    { "Short", Type::Primitive("long") },
    { "Double", Type::Primitive("float") },
};

void WriteSetAttrDirective(const Variable& attr, MethodWriter* writer,
    bool optional) {
  string var_name = optional ? "opts." + attr.name() : attr.name();
  if (attr.type().IsCollection()) {
    const Type& type = attr.type().params().front();
    std::map<string, Type>::const_iterator it =
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
    Type type = attr.type();
    *writer << "opBuilder.setAttr(\"" << attr.name() << "\", ";
    if (type == Type::Class("Class")) {
      *writer << "DataType.fromClass(" << attr.name() << "));" << endl;
    } else {
      *writer << var_name << ");" << endl;
    }
  }
}

}  // namespace

OpTemplate::OpTemplate(const string& op_name, const Type& op_class) :
    op_name_(op_name), op_class_(op_class) {
  imports_.insert({
    Type::Class("Operation", "org.tensorflow"),
    Type::Class("OperationBuilder", "org.tensorflow"),
    Type::Class("Scope", "org.tensorflow.op"),
  });
}

void OpTemplate::AddInput(const Variable& input) {
  AddVariableiable(input, &inputs_);
  if (input.type().IsCollection()) {
    imports_.insert(Type::Class("Operands", "org.tensorflow.op"));
  }
}

void OpTemplate::AddOutput(const Variable& output) {
  AddVariableiable(output, &outputs_);
  if (output.type().IsCollection()) {
    imports_.insert(Type::Class("Arrays", "java.util"));
  }
}

void OpTemplate::CollectImports(const Type& type) {
  auto import_scanner = [this](const Type& type) {
    if (!type.package().empty()) {
      this->imports_.insert(type);
    }
  };
  ScanForTypes(type, &import_scanner);
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
  Type op_class(op_class_);
  op_class.supertype(Type::Class("PrimitiveOp", "org.tensorflow.op"));

  RenderMode mode = DEFAULT;
  if (outputs_.size() == 1) {
      const Variable& output = outputs_.front();
      Type single_type = FindOutputTensorType(output.type());
      if (single_type.IsWildcard()) {
        single_type = Type::Class("Object");
      }
      Type operand = Type::Interface("Operand", "org.tensorflow");
      operand.param(single_type);

      if (output.type().IsCollection()) {
        mode = SINGLE_LIST_OUTPUT;
        op_class.supertype(Type::IterableOf(operand));
        imports_.insert(Type::Interface("Iterator", "java.util"));

      } else {
        mode = SINGLE_OUTPUT;
        op_class.supertype(operand);
        imports_.insert(Type::Class("Output", "org.tensorflow"));
      }
  }
  CollectImports(op_class);

  // Render the op class to the selected target
  Writer writer(src_writer);
  writer << Snippet(io::JoinPath(kGenResourcePath, "licence.snippet.java"));
  ClassWriter* op_writer =
      writer.BeginClass(op_class, imports_, PUBLIC|FINAL);
  if (!opt_attrs_.empty()) {
    RenderOptionsClass(op_writer);
  }
  RenderFactoryMethod(op_writer);
  RenderMethods(op_writer, mode);
  op_writer->WriteFields(outputs_, PRIVATE);
  RenderConstructor(op_writer);
  op_writer->EndClass();
}

void OpTemplate::RenderOptionsClass(ClassWriter* op_writer) {
    Type opt_class = Type::Class("Options");
    opt_class.descr("Class holding optional attributes of this operation");

    ClassWriter* opt_writer =
        op_writer->BeginInnerClass(opt_class, PUBLIC|STATIC);

    std::vector<Variable>::const_iterator var;
    for (var = opt_attrs_.begin(); var != opt_attrs_.end(); ++var) {
      Method setter = Method::Of(var->name(), opt_class);
      setter.arg(*var);

      MethodWriter* set_writer = opt_writer->BeginMethod(setter, PUBLIC);
      *set_writer << "this." << var->name() << " = " << var->name()
          << ";" << endl
          << "return this;" << endl;
      set_writer->EndMethod();
    }
    opt_writer->WriteFields(opt_attrs_, PRIVATE);

    Method opt_ctor = Method::ConstructorFor(opt_class);
    opt_writer->BeginMethod(opt_ctor, PRIVATE)->EndMethod();
    opt_writer->EndClass();
}

void OpTemplate::RenderFactoryMethod(ClassWriter* op_writer) {
  Variable scope =
      Variable::Of("scope", Type::Class("Scope", "org.tensorflow.op"));
  scope.descr("Current graph scope");

  Method factory = Method::Of("create", op_class_);
  factory.descr("Factory method to create a class to wrap a new "
          + op_name_ + " operation to the graph.");
  factory.ret_descr("a new instance of " + op_class_.name());
  factory.arg(scope);
  factory.args(inputs_);
  factory.args(attrs_);
  if (!opt_attrs_.empty()) {
    Variable options = Variable::VarArg("options", Type::Class("Options"));
    options.descr("an object holding optional attributes values");
    factory.arg(options);
  }
  MethodWriter* fct_writer = op_writer->BeginMethod(factory, PUBLIC|STATIC);
  *fct_writer << "OperationBuilder opBuilder = scope.graph().opBuilder(\""
      << op_name_ << "\", scope.makeOpName(\"" << op_name_ << "\"));"
      << endl;
  std::vector<Variable>::const_iterator var;
  for (var = inputs_.begin(); var != inputs_.end(); ++var) {
    if (var->type().IsCollection()) {
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

void OpTemplate::RenderMethods(ClassWriter* op_writer, RenderMode mode) {
  std::vector<Variable>::const_iterator var;

  // Options setters
  for (var = opt_attrs_.begin(); var != opt_attrs_.end(); ++var) {
    Method setter = Method::Of(var->name(), Type::Class("Options"));
    setter.arg(*var);
    MethodWriter* set_writer = op_writer->BeginMethod(setter, PUBLIC|STATIC);
    *set_writer << "return new Options()." << var->name() << "("
            << var->name() << ");" << endl;
    set_writer->EndMethod();
  }
  // Output getters
  for (var = outputs_.begin(); var != outputs_.end(); ++var) {
    Method getter = Method::Of(var->name(), var->type());
    getter.descr(var->descr());
    MethodWriter* get_writer = op_writer->BeginMethod(getter, PUBLIC);
    *get_writer << "return " << var->name() << ";" << endl;
    get_writer->EndMethod();
  }
  // Interface methods
  if (mode == SINGLE_OUTPUT) {
    const Type& output_type = FindOutputTensorType(outputs_.front().type());
    Type return_type = Type::Class("Output", "org.tensorflow")
        .param(output_type);
    Method as_output = Method::Of("asOutput", return_type)
        .annotation(Annotation::Of("Override"));
    // cast the output if not of the same tensor type
    Variable output = outputs_.front();
    bool cast = output_type != FindOutputTensorType(output.type());
    if (cast) {
      as_output.annotation(
          Annotation::Of("SuppressWarnings").attrs("\"unchecked\""));
    }
    MethodWriter* out_writer = op_writer->BeginMethod(as_output, PUBLIC);
    *out_writer << "return ";
    if (cast) {
      *out_writer << "(" << return_type << ") ";
    }
    *out_writer << output.name() << ";" << endl;
    out_writer->EndMethod();

  } else if (mode == SINGLE_LIST_OUTPUT) {
    const Type& output_type = FindOutputTensorType(outputs_.front().type());
    Type operand = Type::Interface("Operand", "org.tensorflow")
        .param(output_type);
    Type return_type = Type::Interface("Iterator", "java.util");
    return_type.param(operand);
    Method iterator = Method::Of("iterator", return_type)
        .annotation(Annotation::Of("Override"))
        .annotation(Annotation::Of("SuppressWarnings")
            .attrs("{\"rawtypes\", \"unchecked\"}"));

    // cast the output list using a raw List
    MethodWriter* it_writer = op_writer->BeginMethod(iterator, PUBLIC);
    *it_writer << "return (" << return_type.name() + ") "
            << outputs_.front().name() << ".iterator();" << endl;
    it_writer->EndMethod();
  }
}

void OpTemplate::RenderConstructor(ClassWriter* op_writer) {
  Variable operation = Variable::Of("operation",
      Type::Class("Operation", "org.tensorflow"));

  Method constructor = Method::ConstructorFor(op_class_).arg(operation);
  constructor.annotation(
      Annotation::Of("SuppressWarnings").attrs("\"unchecked\""));  // FIXME not always required!

  MethodWriter* ctr_writer = op_writer->BeginMethod(constructor, PRIVATE);
  *ctr_writer << "super(operation);" << endl
      << "int outputIdx = 0;" << endl;

  std::vector<Variable>::const_iterator var;
  for (var = outputs_.begin(); var != outputs_.end(); ++var) {
    if (var->type().IsCollection()) {
      string var_length_name = var->name() + "Length";
      *ctr_writer << "int " << var_length_name
          << " = operation.outputListLength(\"" << var->name() << "\");"
          << endl
          << var->name() << " = Arrays.asList(";
      const Type& tensor_type = FindOutputTensorType(var->type());
      if (!tensor_type.IsWildcard()) {
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
