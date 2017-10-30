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

#include "tensorflow/java/src/gen/cc/op_template.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"
#include "tensorflow/java/src/gen/cc/java_writer.h"

namespace tensorflow {
namespace java {

OpTemplate::OpTemplate(const string& op_name) : op_name_(op_name) {
  // Import types we already know of
  imports_.insert({
    Java::Class("PrimitiveOp", "org.tensorflow.op"),
    Java::Class("Operation", "org.tensorflow"),
    Java::Class("OperationBuilder", "org.tensorflow"),
    Java::Class("Scope", "org.tensorflow.op"),
  });
}

void OpTemplate::RenderTo(WritableFile* file) {
  SourceFileWriter src_writer(file);
  Render(&src_writer);
}

void OpTemplate::RenderTo(string* buffer) {
  SourceBufferWriter src_writer(buffer);
  Render(&src_writer);
}

void OpTemplate::Render(SourceWriter* src_writer) {
  RenderMode mode = DEFAULT;
  if (outputs_.size() == 1) {
    mode = IsList(outputs_.front()) ? SINGLE_LIST_OUTPUT : SINGLE_OUTPUT;
  }
  JavaType op_class(op_class_);  // the effective op class
  op_class.supertype(Java::Class("PrimitiveOp", "org.tensorflow.op"));
  JavaType tensor_type;
  switch (mode) {
    case SINGLE_OUTPUT: {
      JavaType output_type = outputs_.front().type();
      tensor_type = output_type.params().front();
      if (Java::IsWildcard(tensor_type)) {
        tensor_type = Java::Class("Object");
      }
      JavaType operand = Java::Interface("Operand", "org.tensorflow");
      operand.param(tensor_type);
      op_class.supertype(operand);
      break;
    }
    case SINGLE_LIST_OUTPUT: {
      JavaType output_type = outputs_.front().type();
      tensor_type = output_type.params().front().params().front();
      if (Java::IsWildcard(tensor_type)) {
        tensor_type = Java::Class("Object");
      }
      JavaType operand = Java::Interface("Operand", "org.tensorflow");
      operand.param(tensor_type);
      op_class.supertype(Java::IterableOf(operand));
      imports_.insert(Java::Interface("Iterator", "java.util"));
      break;
    }
    default: {
      break;
    }
  }
  CollectImports(op_class);
  JavaWriter writer(src_writer);
  writer.WriteSnippet(
      io::JoinPath(kGenResourcePath, "licence.snippet.java"));
  JavaClassWriter* op_writer =
      writer.BeginClass(op_class, imports_, PUBLIC|FINAL);
  bool has_options = !opt_attrs_.empty();
  if (has_options) {
    RenderOptionsClass(op_writer);
  }
  RenderFactoryMethod(op_writer, false);
  if (has_options) {
    RenderFactoryMethod(op_writer, true);
  }
  RenderMethods(op_writer, mode, tensor_type);
  op_writer->WriteFields(outputs_, PRIVATE);
  RenderConstructor(op_writer);
  op_writer->EndOfClass();
}

void OpTemplate::RenderOptionsClass(JavaClassWriter* op_writer) {
    JavaType opt_class = Java::Class("Options");
    opt_class.doc_ptr()
        ->brief("Class holding optional attributes of this operation");

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

void OpTemplate::RenderFactoryMethod(JavaClassWriter* op_writer,
    bool with_options) {

  JavaVar scope = Java::Var("scope", Java::Class("Scope", "org.tensorflow.op"));
  scope.doc_ptr()->brief("Current graph scope");

  JavaMethod factory = Java::Method("create", op_class_);
  factory.doc_ptr()->brief("Factory method to create a class to wrap a new "
          + op_name_ + " operation to the graph.");
  factory.doc_ptr()->value("a new instance of " + op_class_.name());
  factory.arg(scope);
  factory.args(inputs_);
  factory.args(attrs_);

  // For each output variable whose generic is not inferred by another operand,
  // we require the user to pass its type explicitly
  std::set<string> inferred_generic_names;
  GenericTypeScanner inputs_scanner(&inferred_generic_names);
  factory.ScanTypes(&inputs_scanner, true);

  std::vector<JavaVar>::const_iterator var;
  for (var = outputs_.begin(); var != outputs_.end(); ++var) {
    GenericTypeScanner output_scanner(&inferred_generic_names);
    var->type().Scan(&output_scanner);

    std::vector<const JavaType*>::const_iterator generic_it;
    for (generic_it = output_scanner.discoveredTypes().cbegin();
        generic_it != output_scanner.discoveredTypes().cend(); ++generic_it) {
      JavaVar output_class = Java::Var(var->name(), Java::ClassOf(**generic_it));
      factory.arg(output_class);
    }
  }

  if (with_options) {
    JavaVar options = Java::Var("options", Java::Class("Options"));
    options.doc_ptr()->brief("an object holding optional attributes values");
    factory.arg(options);
  }

  JavaMethodWriter* factory_writer =
      op_writer->BeginMethod(factory, PUBLIC|STATIC);

  factory_writer->WriteLine(string("OperationBuilder opBuilder = ")
      + "scope.graph().opBuilder(\"" + op_name_
      + "\", scope.makeOpName(\"" + op_name_ + "\"));");

  for (var = inputs_.begin(); var != inputs_.end(); ++var) {
    if (IsList(*var)) {
      factory_writer->WriteLine(
          "opBuilder.addInputList(Operands.asOutputs(" + var->name() + "));");
    } else {
      factory_writer->WriteLine(
          "opBuilder.addInput(" + var->name() + ".asOutput());");
    }
  }
  for (var = attrs_.begin(); var != attrs_.end(); ++var) {
    factory_writer->WriteLine(
        "opBuilder.setAttr(\"" + var->name() + "\", " + var->name() + ");");
  }
  if (with_options) {
    for (var = opt_attrs_.begin(); var != opt_attrs_.end(); ++var) {
      factory_writer->BeginBlock("if (options." + var->name() + " != null)")
          ->WriteLine("opBuilder.setAttr(\"" + var->name()
              + "\", options." + var->name() + ");")
          ->EndOfBlock();
    }
  }
  factory_writer->Write("return new ")
      ->Write(op_class_)
      ->WriteLine("(opBuilder.build());")
      ->EndOfMethod();
}

void OpTemplate::RenderMethods(JavaClassWriter* op_writer, RenderMode mode,
    const JavaType& output_tensor_type) {
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
    JavaType return_type = Java::Class("Output", "org.tensorflow");
    return_type.param(output_tensor_type);
    JavaMethod as_output = Java::Method("asOutput", return_type);
    as_output.annotation(Java::Annot("Override"));
    JavaMethodWriter* method_writer =
        op_writer->BeginMethod(as_output, PUBLIC)->Write("return ");

    // cast the output if not of the same tensor type
    JavaVar output = outputs_.front();
    if (output_tensor_type != output.type().params().front()) {
      method_writer->Write("(")->Write(return_type)->Write(") ");
    }
    method_writer->WriteLine(output.name() + ";")->EndOfMethod();

  } else if (mode == SINGLE_LIST_OUTPUT) {
    JavaType operand = Java::Interface("Operand", "org.tensorflow");
    operand.param(output_tensor_type);
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

  JavaMethodWriter* ctor_writer = op_writer->BeginMethod(ctor, PRIVATE);
  ctor_writer->WriteLine("super(operation);")
      ->WriteLine("int outputIdx = 0;");

  std::vector<JavaVar>::const_iterator var;
  for (var = outputs_.begin(); var != outputs_.end(); ++var) {
    if (IsList(*var)) {
      string var_length_name = var->name() + "Length";
      ctor_writer->WriteLine("int " + var_length_name
          + " = operation.outputListLength(\"" + var->name() + "\");");

      // output lists must be cast explicitly
      ctor_writer->Write(var->name() + " = Arrays.asList((")
          ->Write(var->type().params().front())
          ->Write("[]) operation.outputList(outputIdx, ")
          ->WriteLine(var_length_name + "));");
      ctor_writer->WriteLine("outputIdx += " + var_length_name + ";");

    } else {
      ctor_writer->WriteLine(var->name() + " = operation.output(outputIdx++);");
    }
  }
  ctor_writer->EndOfMethod();
}

void OpTemplate::CollectImports(const JavaType& type) {
  auto import_scanner = [this](const JavaType* type) {
    if (!type->package().empty()) {
      this->imports_.insert(*type);
    }
  };
  type.Scan(&import_scanner);
}

}  // namespace java
}  // namespace tensorflow
