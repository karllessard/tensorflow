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

#include "tensorflow/java/src/gen/cc/op_template.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"
#include "tensorflow/java/src/gen/cc/java_writer.h"

namespace tensorflow {

OpTemplate::OpTemplate(const string& op_name, const string& op_group)
  : op_name(op_name), op_group(op_group) {
  // Register imports of class dependencies we already know of
  imports.insert({
    "org.tensorflow.op.PrimitiveOp",
    "org.tensorflow.Operation",
    "org.tensorflow.OperationBuilder",
    "org.tensorflow.op.Scope",
    "org.tensorflow.op.Operands",
    "org.tensorflow.op.annotation.Operator"
  });
}

void OpTemplate::Render(SourceOutputStream* stream) {
  RenderMode mode = SelectRenderMode();
  switch (mode) {
  case SINGLE_OUTPUT: {
    imports.insert({
      "org.tensorflow.Output",
      "org.tensorflow.Operand"
    });
    op_class.interface(java::Type("Operand", "org.tensorflow"));
    break;
  }
  case SINGLE_LIST_OUTPUT: {
    imports.insert({
      "java.util.Iterator",
      "org.tensorflow.Operand"
    });
    java::Type iterable("Iterable");
    iterable.param(java::Type("Operand", "org.tensorflow"));
    op_class.interface(iterable);
    break;
  }
  default:
    break;
  }
  op_class.supertype(java::Type("PrimitiveOp", "org.tensorflow.op"));

  java::Writer writer(stream);
  string licence_file = io::JoinPath(java::kGenResourcePath,
      "licence.snippet.java");
  writer.WriteSnippet(licence_file);
  java::ClassWriter* op_writer =
      writer.BeginClass(op_class, imports, java::PUBLIC|java::FINAL);
  bool has_options = !opt_attrs.empty();
  if (has_options) {
    RenderOptionsClass(op_writer);
  }
  RenderFactoryMethod(op_writer, false);
  if (has_options) {
    RenderFactoryMethod(op_writer, true);
  }
  RenderMethods(op_writer, mode);
  op_writer->WriteFields(outputs, java::PRIVATE);
  RenderConstructor(op_writer);
  op_writer->EndOfClass();
}

OpTemplate::RenderMode OpTemplate::SelectRenderMode() {
  if (outputs.size() == 1) {
    return IsList(outputs.front()) ? SINGLE_LIST_OUTPUT : SINGLE_OUTPUT;
  }
  return DEFAULT;
}

void OpTemplate::RenderOptionsClass(java::ClassWriter* op_writer) {
    java::Class opt_class("Options");
    opt_class.doc_ptr()
        ->brief("Class holding optional attributes of this operation");

    java::ClassWriter* opt_writer =
        op_writer->BeginInnerClass(opt_class, java::PUBLIC|java::STATIC);

    std::list<java::Variable>::const_iterator var;
    for (var = opt_attrs.begin(); var != opt_attrs.end(); ++var) {
      java::Method setter(var->name(), opt_class);
      setter.arg(*var);

      opt_writer->BeginMethod(setter, java::PUBLIC)
                ->WriteLine("this." + var->name() + " = " + var->name() + ";")
                ->WriteLine("return this;")
                ->EndOfMethod();
    }
    opt_writer->WriteFields(opt_attrs, java::PRIVATE);

    java::Method constructor(opt_class.name());
    opt_writer->BeginMethod(constructor, java::PRIVATE)->EndOfMethod();
    opt_writer->EndOfClass();
}

void OpTemplate::RenderFactoryMethod(java::ClassWriter* op_writer,
    bool with_options) {

  java::Variable scope("scope", java::Type("Scope", "org.tensorflow.op"));
  scope.doc_ptr()->brief("Current graph scope");

  java::Method factory("create", op_class);
  factory.doc_ptr()
      ->brief("Factory method to create a class to wrap a new "
          + op_name + " operation to the graph.")
      ->returnValue("a new instance of " + op_class.name());
  factory.arg(scope)->args(inputs)->args(attrs);

  if (with_options) {
    java::Variable options("options", java::Type("Options"));
    options.doc_ptr()->brief("an object holding optional attributes values");
    factory.arg(options);
  }

  java::MethodWriter* factory_writer =
      op_writer->BeginMethod(factory, java::PUBLIC|java::STATIC);

  factory_writer->WriteLine(string("OperationBuilder opBuilder = ")
      + "scope.graph().opBuilder(\"" + op_name
      + "\", scope.makeOpName(\"" + op_name + "\"));");

  std::list<java::Variable>::const_iterator var;
  for (var = inputs.begin(); var != inputs.end(); ++var) {
    if (IsList(*var)) {
      factory_writer->WriteLine(
          "opBuilder.addInputList(Operands.asOutputs(" + var->name() + "));");
    } else {
      factory_writer->WriteLine(
          "opBuilder.addInput(" + var->name() + ".asOutput());");
    }
  }
  for (var = attrs.begin(); var != attrs.end(); ++var) {
    factory_writer->WriteLine(
        "opBuilder.setAttr(\"" + var->name() + "\", " + var->name() + ");");
  }
  if (with_options) {
    for (var = opt_attrs.begin(); var != opt_attrs.end(); ++var) {
      factory_writer->BeginBlock("if (options." + var->name() + " != null)")
                    ->WriteLine("opBuilder.setAttr(\"" + var->name()
                        + "\", options." + var->name() + ");")
                    ->EndOfBlock();
    }
  }
  factory_writer->WriteLine("return new " + op_class.name()
      + "(opBuilder.build());");
  factory_writer->EndOfMethod();
}

void OpTemplate::RenderMethods(java::ClassWriter* op_writer, RenderMode mode) {
  std::list<java::Variable>::const_iterator var;
  // Options setters
  for (var = opt_attrs.begin(); var != opt_attrs.end(); ++var) {
    java::Method setter(var->name(), java::Type("Options"));
    setter.arg(*var);

    op_writer->BeginMethod(setter, java::PUBLIC|java::STATIC)
             ->WriteLine("return new Options()." + var->name() + "("
                 + var->name() + ");")
             ->EndOfMethod();
  }
  // Output getters
  for (var = outputs.begin(); var != outputs.end(); ++var) {
    java::Method getter(var->name(), var->type());
    getter.doc(var->doc());

    op_writer->BeginMethod(getter, java::PUBLIC)
             ->WriteLine("return " + var->name() + ";")
             ->EndOfMethod();
  }
  // Implemented methods
  if (mode == SINGLE_OUTPUT) {
    java::Method as_output("asOutput", java::Type("Output", "org.tensorflow"));
    as_output.annotation(java::Annotation("Override"));

    op_writer->BeginMethod(as_output, java::PUBLIC)
             ->WriteLine("return " + outputs.front().name() + ";")
             ->EndOfMethod();

  } else if (mode == SINGLE_LIST_OUTPUT) {
    java::Type return_type("Iterator", "java.util");
    return_type.param(java::Type("Operand", "org.tensorflow"));
    java::Method iterator("iterator", return_type);
    iterator.annotation(java::Annotation("Override"));
    java::Annotation suppressWarnings("SuppressWarnings");
    suppressWarnings.attrs("{\"rawtypes\", \"unchecked\"}");
    iterator.annotation(suppressWarnings);

    op_writer->BeginMethod(iterator, java::PUBLIC)
             ->WriteLine("return (Iterator)" + outputs.front().name()
                 + ".iterator();")
             ->EndOfMethod();
  }
}

void OpTemplate::RenderConstructor(java::ClassWriter* op_writer) {
  java::Variable operation("operation",
      java::Type("Operation", "org.tensorflow"));
  java::Method ctor(op_class.name());
  ctor.arg(operation);

  java::MethodWriter* ctor_writer = op_writer->BeginMethod(ctor, java::PRIVATE);
  ctor_writer->WriteLine("super(operation);")
             ->WriteLine("int outputIdx = 0;");

  std::list<java::Variable>::const_iterator var;
  for (var = outputs.begin(); var != outputs.end(); ++var) {
    if (IsList(*var)) {
      string var_length_name = var->name() + "Length";
      ctor_writer->WriteLine("int " + var_length_name
          + " = operation.outputListLength(\"" + var->name() + "\");");
      ctor_writer->WriteLine(var->name()
          + " = Arrays.asList(operation.outputList(outputIdx, "
          + var_length_name + "));");
      ctor_writer->WriteLine("outputIdx += " + var_length_name + ";");

    } else {
      ctor_writer->WriteLine(var->name() + " = operation.output(outputIdx++);");
    }
  }
  ctor_writer->EndOfMethod();
}

void OpTemplate::CollectImports(const java::Type& type) {
  auto import_scanner = [this](const java::Type* type) {
    if (!type->package().empty() && type->package() !=
        this->op_class.package()) {
      this->imports.insert(type->package() + "." + type->name());
    }
  };
  type.Accept(&import_scanner);
}

} /* namespace tensorflow */
