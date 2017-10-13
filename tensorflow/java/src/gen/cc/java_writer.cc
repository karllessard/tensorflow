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

#include <set>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"
#include "tensorflow/java/src/gen/cc/java_writer.h"

namespace tensorflow {
namespace java {
namespace {

class GenericTypeScanner {
 public:
  explicit GenericTypeScanner(std::set<string>* declared_names)
    : declared_names_(declared_names) {}
  const std::list<const Type*>& discoveredTypes() const {
    return discovered_types_;
  }
  void operator()(const Type* type) {
    if (type->generic() && !type->name().empty()
        && (declared_names_->find(type->name()) == declared_names_->end())) {
      discovered_types_.push_back(type);
      declared_names_->insert(type->name());
    }
  }

 private:
  std::list<const Type*> discovered_types_;
  std::set<string>* declared_names_;
};

void WriteModifiers(int modifiers, SourceOutputStream* stream) {
  if (modifiers & PUBLIC) {
    stream->Append("public ");
  } else if (modifiers & PROTECTED) {
    stream->Append("protected ");
  } else if (modifiers & PRIVATE) {
    stream->Append("private ");
  }
  if (modifiers & STATIC) {
    stream->Append("static ");
  }
  if (modifiers & FINAL) {
    stream->Append("final ");
  }
}

void WriteType(const Type& type, SourceOutputStream* stream) {
  if (type.generic()) {
    stream->Append(type.name().empty() ? "?" : type.name());
  } else {
    stream->Append(type.name());
    if (!type.params().empty()) {
      stream->Append("<");
      std::list<Type>::const_iterator it;
      for (it = type.params().cbegin(); it != type.params().cend(); ++it) {
        if (it != type.params().cbegin()) {
          stream->Append(", ");
        }
        WriteType(*it, stream);
      }
      stream->Append(">");
    }
  }
}

void WriteGenerics(const std::list<const Type*>& generics,
    SourceOutputStream* stream) {
  stream->Append("<");
  for (std::list<const Type*>::const_iterator it = generics.cbegin();
      it != generics.cend(); ++it) {
    stream->Append((*it)->name());
    if ((*it)->supertype_ptr() != nullptr) {
      stream->Append(" extends ")->Append((*it)->supertype_ptr()->name());
    }
  }
  stream->Append("> ");
}

void WriteAnnotations(const std::list<Annotation>& annotations,
    SourceOutputStream* stream) {
  std::list<Annotation>::const_iterator it;
  for (it = annotations.cbegin(); it != annotations.cend(); ++it) {
    stream->Append("@" + it->name());
    if (!it->attrs().empty()) {
      stream->Append("(")->Append(it->attrs())->Append(")");
    }
    stream->EndOfLine();
  }
}

void WriteDoc(const Doc& doc, const std::list<Variable>* params,
    SourceOutputStream* stream) {
  stream->Append("/**")->EndOfLine()->Prefix(" * ");
  if (!doc.brief().empty()) {
    stream->Inline(doc.brief())->EndOfLine();
  }
  if (!doc.description().empty()) {
    stream->Append("<p>")
          ->EndOfLine()
          ->Inline(doc.description())
          ->EndOfLine();
  }
  if (params != NULL) {
    stream->EndOfLine();
    std::list<Variable>::const_iterator it;
    for (it = params->begin(); it != params->end(); ++it) {
      stream->Append("@param ")
            ->Append(it->name())
            ->Append(" ")
            ->Inline(it->doc().brief())
            ->EndOfLine();
    }
  }
  if (!doc.returnValue().empty()) {
    stream->Inline("@return " + doc.returnValue())->EndOfLine();
  }
  stream->RemovePrefix()->Append(" **/")->EndOfLine();
}

}  // namespace

ClassWriter* ClassWriter::Begin(const Class& clazz, int modifiers) {
  GenericTypeScanner generics(&declared_generics_names_);
  clazz.Accept(&generics);
  WriteDoc(clazz.doc(), nullptr, stream_);
  if (!clazz.annotations().empty()) {
    WriteAnnotations(clazz.annotations(), stream_);
  }
  WriteModifiers(modifiers, stream_);
  stream_->Append("class ")->Append(clazz.name());
  if (!generics.discoveredTypes().empty()) {
    WriteGenerics(generics.discoveredTypes(), stream_);
  }
  if (clazz.supertype_ptr() != nullptr) {
    stream_->Append(" extends ");
    WriteType(*clazz.supertype_ptr(), stream_);
  }
  for (std::list<Type>::const_iterator it = clazz.interfaces().cbegin();
      it != clazz.interfaces().cend(); ++it) {
    stream_->Append(it == clazz.interfaces().cbegin() ? " implements " : ", ");
    WriteType(*it, stream_);
  }
  BaseWriter::BeginBlock();
  return this;
}

ClassWriter* ClassWriter::WriteFields(
    const std::list<Variable>& fields, int modifiers) {
  stream_->EndOfLine();
  std::list<Variable>::const_iterator it;
  for (it = fields.cbegin(); it != fields.cend(); ++it) {
    WriteModifiers(modifiers, stream_);
    WriteType(it->type(), stream_);
    stream_->Append(" ")->Append(it->name())->Append(";")->EndOfLine();
  }
  return this;
}

MethodWriter* ClassWriter::BeginMethod(const Method& method,
    int modifiers) {
  stream_->EndOfLine();
  WriteDoc(method.doc(), &method.args(), stream_);
  if (!method.annotations().empty()) {
    WriteAnnotations(method.annotations(), stream_);
  }
  MethodWriter* method_writer;
  if (modifiers & STATIC) {
    method_writer = new MethodWriter(stream_);
  } else {
    method_writer = new MethodWriter(stream_, declared_generics_names_);
  }
  return method_writer->Begin(method, modifiers);
}

MethodWriter* MethodWriter::Begin(const Method& method,
    int modifiers) {
  GenericTypeScanner generics(&declared_generics_names_);
  method.Accept(&generics);
  WriteModifiers(modifiers, stream_);
  if (!generics.discoveredTypes().empty()) {
    WriteGenerics(generics.discoveredTypes(), stream_);
  }
  if (method.type().valid()) {
    WriteType(method.type(), stream_);
    stream_->Append(" ");
  }
  stream_->Append(method.name())->Append("(");
  if (!method.args().empty()) {
    for (std::list<Variable>::const_iterator arg = method.args().cbegin();
        arg != method.args().cend(); ++arg) {
      if (arg != method.args().cbegin()) {
        stream_->Append(", ");
      }
      WriteType(arg->type(), stream_);
      stream_->Append(" ")->Append(arg->name());
    }
  }
  stream_->Append(")");
  BaseWriter::BeginBlock();
  return this;
}

ClassWriter* ClassWriter::BeginInnerClass(const Class& clazz,
    int modifiers) {
  stream_->EndOfLine();
  ClassWriter* class_writer;
  if (modifiers & STATIC) {
    class_writer = new ClassWriter(stream_);
  } else {
    class_writer = new ClassWriter(stream_, declared_generics_names_);
  }
  return class_writer->Begin(clazz, modifiers);
}

ClassWriter* Writer::BeginClass(const Class& clazz,
    const std::set<string>& imports, int modifiers) {
  WriteLine("package " + clazz.package() + ";");
  stream_->EndOfLine();
  std::set<string>::const_iterator it;
  for (it = imports.cbegin(); it != imports.cend(); ++it) {
    WriteLine("import " + *it + ";");
  }
  stream_->EndOfLine();
  ClassWriter* class_writer = new ClassWriter(stream_);
  return class_writer->Begin(clazz, modifiers);
}

}  // namespace java
}  // namespace tensorflow
