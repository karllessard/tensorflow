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
#include <set>
#include <vector>
#include <deque>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/java/src/gen/cc/source_writer.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"
#include "tensorflow/java/src/gen/cc/java_writer.h"

namespace tensorflow {
namespace java {
namespace {

/// \brief A function used to collect generic type parameters discovered while
///        scanning an object for types (e.g. Method::ScanTypes)
class GenericTypeScanner {
 public:
  explicit GenericTypeScanner(std::set<string>* declared_names)
    : declared_names_(declared_names) {}
  const std::vector<const Type*>& discoveredTypes() const {
    return discovered_types_;
  }
  void operator()(const Type* type) {
    if (type->kind() == Type::GENERIC && !type->name().empty()
        && (declared_names_->find(type->name()) == declared_names_->end())) {
      discovered_types_.push_back(type);
      declared_names_->insert(type->name());
    }
  }
 private:
  std::vector<const Type*> discovered_types_;
  std::set<string>* declared_names_;
};

void WriteModifiers(int modifiers, SourceWriter* src_writer) {
  if (modifiers & PUBLIC) {
    src_writer->Write("public ");
  } else if (modifiers & PROTECTED) {
    src_writer->Write("protected ");
  } else if (modifiers & PRIVATE) {
    src_writer->Write("private ");
  }
  if (modifiers & STATIC) {
    src_writer->Write("static ");
  }
  if (modifiers & FINAL) {
    src_writer->Write("final ");
  }
}

void WriteType(const Type& type, SourceWriter* src_writer) {
  if (type.kind() == Type::Kind::GENERIC && type.name().empty()) {
    src_writer->Write("?");
  } else {
    src_writer->Write(type.name());
  }
  if (!type.params().empty()) {
    src_writer->Write("<");
    std::vector<Type>::const_iterator it;
    for (it = type.params().cbegin(); it != type.params().cend(); ++it) {
      if (it != type.params().cbegin()) {
        src_writer->Write(", ");
      }
      WriteType(*it, src_writer);
    }
    src_writer->Write(">");
  }
}

void WriteGenerics(const std::vector<const Type*>& generics,
    SourceWriter* src_writer) {
  src_writer->Write("<");
  for (std::vector<const Type*>::const_iterator it = generics.cbegin();
      it != generics.cend(); ++it) {
    if (it != generics.cbegin()) {
      src_writer->Write(", ");
    }
    src_writer->Write((*it)->name());
    if (!(*it)->supertypes().empty()) {
      src_writer->Write(" extends ");
      WriteType((*it)->supertypes().front(), src_writer);
    }
  }
  src_writer->Write(">");
}

void WriteAnnotationations(const std::vector<Annotation>& annotations,
    SourceWriter* src_writer) {
  std::vector<Annotation>::const_iterator it;
  for (it = annotations.cbegin(); it != annotations.cend(); ++it) {
    src_writer->Write("@" + it->name());
    if (!it->attrs().empty()) {
      src_writer->Write("(")->Write(it->attrs())->Write(")");
    }
    src_writer->EndOfLine();
  }
}

void WriteDoc(const string& descr, const string* ret_descr,
    const std::vector<Variable>* params, SourceWriter* src_writer) {
  if (descr.empty() && (ret_descr == nullptr || ret_descr->empty())
      && (params == nullptr || params->empty())) {
    return;  // no doc to write
  }
  bool line_break = false;
  src_writer->Write("/**")->EndOfLine()->LinePrefix(" * ");
  if (!descr.empty()) {
    src_writer->Inline(descr)->EndOfLine();
    line_break = true;
  }
  if (params != nullptr && !params->empty()) {
    if (line_break) {
      src_writer->EndOfLine();
      line_break = false;
    }
    std::vector<Variable>::const_iterator it;
    for (it = params->begin(); it != params->end(); ++it) {
      src_writer->Write("@param ")->Write(it->name());
      if (!it->descr().empty()) {
        src_writer->Write(" ")->Inline(it->descr());
      }
      src_writer->EndOfLine();
    }
  }
  if (ret_descr != nullptr && !ret_descr->empty()) {
    if (line_break) {
      src_writer->EndOfLine();
    }
    src_writer->Write("@return ")->Inline(*ret_descr)->EndOfLine();
  }
  src_writer->RemoveLinePrefix()->Write(" **/")->EndOfLine();
}

}  // namespace

SourceStream& SourceStream::operator<<(const Type& type) {
  WriteType(type, src_writer_);
  return *this;
}

ClassWriter* ClassWriter::Begin(const Type& clazz, int modifiers) {
  GenericTypeScanner generics(&declared_generics_);
  ScanForTypes(clazz, &generics);
  WriteDoc(clazz.descr(), nullptr, nullptr, src_writer_);
  if (!clazz.annotations().empty()) {
    WriteAnnotationations(clazz.annotations(), src_writer_);
  }
  WriteModifiers(modifiers, src_writer_);
  *this << "class " << clazz.name();
  if (!generics.discoveredTypes().empty()) {
    WriteGenerics(generics.discoveredTypes(), src_writer_);
  }
  if (!clazz.supertypes().empty()) {
    std::deque<Type>::const_iterator it = clazz.supertypes().cbegin();
    if (it->kind() == Type::CLASS) {  // superclass is always first in list
      *this << " extends " << *it++;
    }
    bool first_inf = true;
    while (it != clazz.supertypes().cend()) {
      *this << (first_inf ? " implements " : ", ") << *it++;
      first_inf = false;
    }
  }
  *this << beginb;
  return this;
}

ClassWriter* ClassWriter::WriteFields(
    const std::vector<Variable>& fields, int modifiers) {
  *this << endl;
  std::vector<Variable>::const_iterator it;
  for (it = fields.cbegin(); it != fields.cend(); ++it) {
    WriteModifiers(modifiers, src_writer_);
    *this << it->type() << " " << it->name() << ";" << endl;
  }
  return this;
}

MethodWriter* ClassWriter::BeginMethod(const Method& method,
    int modifiers) {
  *this << endl;
  const string* ret_descr =
      method.constructor() ? nullptr : &method.ret_descr();
  WriteDoc(method.descr(), ret_descr, &method.args(), src_writer_);
  if (!method.annotations().empty()) {
    WriteAnnotationations(method.annotations(), src_writer_);
  }
  MethodWriter* method_writer;
  if (modifiers & STATIC) {
    method_writer = new MethodWriter(src_writer_);
  } else {
    method_writer = new MethodWriter(src_writer_, declared_generics_);
  }
  return method_writer->Begin(method, modifiers);
}

MethodWriter* MethodWriter::Begin(const Method& method,
    int modifiers) {
  GenericTypeScanner generics(&declared_generics_);
  ScanForTypes(method, &generics);
  WriteModifiers(modifiers, src_writer_);
  if (!generics.discoveredTypes().empty()) {
    WriteGenerics(generics.discoveredTypes(), src_writer_);
    *this << " ";
  }
  if (!method.constructor()) {
    *this << method.ret_type() << " ";
  }
  *this << method.name() << "(";
  if (!method.args().empty()) {
    for (std::vector<Variable>::const_iterator arg = method.args().cbegin();
        arg != method.args().cend(); ++arg) {
      if (arg != method.args().cbegin()) {
        *this << ", ";
      }
      *this << arg->type() << (arg->variadic() ? "... " : " ") << arg->name();
    }
  }
  *this << ")" << beginb;
  return this;
}

ClassWriter* ClassWriter::BeginInnerClass(const Type& clazz,
    int modifiers) {
  *this << endl;
  ClassWriter* class_writer;
  if (modifiers & STATIC) {
    class_writer = new ClassWriter(src_writer_);
  } else {
    class_writer = new ClassWriter(src_writer_, declared_generics_);
  }
  return class_writer->Begin(clazz, modifiers);
}

ClassWriter* Writer::BeginClass(const Type& clazz,
    const std::set<Type>& imports, int modifiers) {
  *this << "package " << clazz.package() << ";" << endl << endl;
  if (!imports.empty()) {
    std::set<Type>::const_iterator it;
    for (it = imports.cbegin(); it != imports.cend(); ++it) {
      if (!it->package().empty() && it->package() != clazz.package()) {
        *this << "import " << it->package() << "." << it->name() << ";" << endl;
      }
    }
    *this << endl;
  }
  ClassWriter* class_writer = new ClassWriter(src_writer_);
  return class_writer->Begin(clazz, modifiers);
}

}  // namespace java
}  // namespace tensorflow
