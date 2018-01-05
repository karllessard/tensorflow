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
#include "tensorflow/java/src/gen/cc/java_defs.h"
#include "tensorflow/java/src/gen/cc/java_writers.h"

namespace tensorflow {
namespace java {
namespace {

// A function used to collect generic type parameters discovered while
// scanning an object for types (see ScanForTypes()).
class GenericTypeScanner {
 public:
  explicit GenericTypeScanner(std::set<string>* declared_names)
    : declared_names_(declared_names) {}
  const std::vector<const Type*>& discoveredTypes() const {
    return discovered_types_;
  }
  void operator()(const Type& type) {
    if (type.kind() == Type::GENERIC && !type.name().empty()
        && (declared_names_->find(type.name()) == declared_names_->end())) {
      discovered_types_.push_back(&type);
      declared_names_->insert(type.name());
    }
  }
 private:
  std::vector<const Type*> discovered_types_;
  std::set<string>* declared_names_;
};

// Writes a bitwise list of Java modifiers.
void WriteModifiers(int modifiers, WriterInterface* writer) {
  if (modifiers & PUBLIC) {
    writer->Append("public ");
  } else if (modifiers & PROTECTED) {
    writer->Append("protected ");
  } else if (modifiers & PRIVATE) {
    writer->Append("private ");
  }
  if (modifiers & STATIC) {
    writer->Append("static ");
  }
  if (modifiers & FINAL) {
    writer->Append("final ");
  }
}

// Writes the signature of a Java type.
void WriteType(const Type& type, WriterInterface* writer) {
  if (type.kind() == Type::Kind::GENERIC && type.name().empty()) {
    writer->Append("?");
  } else {
    writer->Append(type.name());
  }
  if (!type.parameters().empty()) {
    writer->Append("<");
    std::vector<Type>::const_iterator it = type.parameters().cbegin();
    bool first = true;
    while (it != type.parameters().cend()) {
      if (!first) {
        writer->Append(", ");
      } else {
        first = false;
      }
      WriteType(*it, writer);
      ++it;
    }
    writer->Append(">");
  }
}

// Writes the declaration of a list of Java generics.
void WriteGenerics(const std::vector<const Type*>& generics,
    WriterInterface* writer) {
  writer->Append("<");
  for (std::vector<const Type*>::const_iterator it = generics.cbegin();
      it != generics.cend(); ++it) {
    if (it != generics.cbegin()) {
      writer->Append(", ");
    }
    writer->Append((*it)->name());
    if (!(*it)->supertypes().empty()) {
      writer->Append(" extends ");
      WriteType((*it)->supertypes().front(), writer);
    }
  }
  writer->Append(">");
}

// Writes the usage of a Java annotation.
void WriteAnnotations(const std::vector<Annotation>& annotations,
    WriterInterface* writer) {
  std::vector<Annotation>::const_iterator it;
  for (it = annotations.cbegin(); it != annotations.cend(); ++it) {
    writer->Append("@" + it->name());
    if (!it->attributes().empty()) {
      writer->Append("(").Append(it->attributes()).Append(")");
    }
    writer->EndLine();
  }
}

// Writes documentation in the Javadoc format.
void WriteDoc(const string& description, const string* return_description,
    const std::vector<Variable>* parameters, WriterInterface* writer) {
  if (description.empty()
      && (return_description == nullptr || return_description->empty())
      && (parameters == nullptr || parameters->empty())) {
    return;  // no doc to write
  }
  bool line_break = false;
  writer->Append("/**").EndLine().SetLinePrefix(" * ");
  if (!description.empty()) {
    writer->Write(description).EndLine();
    line_break = true;
  }
  if (parameters != nullptr && !parameters->empty()) {
    if (line_break) {
      writer->EndLine();
      line_break = false;
    }
    std::vector<Variable>::const_iterator it;
    for (it = parameters->begin(); it != parameters->end(); ++it) {
      writer->Append("@param ").Append(it->name());
      if (!it->description().empty()) {
        writer->Append(" ").Write(it->description());
      }
      writer->EndLine();
    }
  }
  if (return_description != nullptr && !return_description->empty()) {
    if (line_break) {
      writer->EndLine();
    }
    writer->Append("@return ").Write(*return_description).EndLine();
  }
  writer->UnsetLinePrefix().Append(" **/").EndLine();
}

}  // namespace

void MethodWriter::EndMethod() {
  *this << endb;
  delete this;
}

MethodWriter* MethodWriter::BeginMethod(const Method& method,
    int modifiers) {
  GenericTypeScanner generics(&declared_generics_);
  ScanForTypes(method, &generics);
  WriteModifiers(modifiers, this);
  if (!generics.discoveredTypes().empty()) {
    WriteGenerics(generics.discoveredTypes(), this);
    *this << " ";
  }
  if (!method.constructor()) {
    *this << method.return_type() << " ";
  }
  *this << method.name() << "(";
  if (!method.arguments().empty()) {
    std::vector<Variable>::const_iterator arg = method.arguments().cbegin();
    bool first = true;
    while (arg != method.arguments().cend()) {
      if (!first) {
        *this << ", ";
      } else {
        first = false;
      }
      *this << arg->type() << (arg->variadic() ? "... " : " ") << arg->name();
      ++arg;
    }
  }
  *this << ")" << beginb;
  return this;
}

ClassWriter* ClassWriter::WriteFields(
    const std::vector<Variable>& fields, int modifiers) {
  *this << endl;
  std::vector<Variable>::const_iterator it;
  for (it = fields.cbegin(); it != fields.cend(); ++it) {
    WriteModifiers(modifiers, this);
    *this << it->type() << " " << it->name() << ";" << endl;
  }
  return this;
}

MethodWriter* ClassWriter::BeginMethod(const Method& method,
    int modifiers) {
  *this << endl;
  const string* ret_descr =
      method.constructor() ? nullptr : &method.return_description();
  WriteDoc(method.description(), ret_descr, &method.arguments(), this);
  if (!method.annotations().empty()) {
    WriteAnnotations(method.annotations(), this);
  }
  MethodWriter* method_writer;
  if (modifiers & STATIC) {
    method_writer = new MethodWriter(this);
  } else {
    method_writer = new MethodWriter(this, declared_generics_);
  }
  return method_writer->BeginMethod(method, modifiers);
}

ClassWriter* ClassWriter::BeginInnerClass(const Type& clazz,
    int modifiers) {
  *this << endl;
  ClassWriter* class_writer;
  if (modifiers & STATIC) {
    class_writer = new ClassWriter(this);
  } else {
    class_writer = new ClassWriter(this, declared_generics_);
  }
  return class_writer->BeginClass(clazz, modifiers);
}

void ClassWriter::EndClass() {
  *this << endb;
  delete this;
}

ClassWriter* ClassWriter::BeginClass(const Type& clazz, int modifiers) {
  GenericTypeScanner generics(&declared_generics_);
  ScanForTypes(clazz, &generics);
  WriteDoc(clazz.description(), nullptr, nullptr, this);
  if (!clazz.annotations().empty()) {
    WriteAnnotations(clazz.annotations(), this);
  }
  WriteModifiers(modifiers, this);
  *this << "class " << clazz.name();
  if (!generics.discoveredTypes().empty()) {
    WriteGenerics(generics.discoveredTypes(), this);
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

SourceWriter& SourceWriter::Append(const StringPiece& str) {
  if (!str.empty()) {
    if (newline_) {
      DoAppend(left_margin_ + line_prefix_);
      newline_ = false;
    }
    DoAppend(str);
  }
  return *this;
}

SourceWriter& SourceWriter::Write(const string& str) {
  size_t line_pos = 0;
  do {
    size_t start_pos = line_pos;
    line_pos = str.find('\n', start_pos);
    if (line_pos != string::npos) {
      ++line_pos;
      Append(StringPiece(str.data() + start_pos, line_pos - start_pos));
      newline_ = true;
    } else {
      Append(StringPiece(str.data() + start_pos, str.size() - start_pos));
    }
  } while (line_pos != string::npos && line_pos < str.size());

  return *this;
}

SourceWriter& SourceWriter::EndLine() {
  Append("\n");
  newline_ = true;
  return *this;
}

SourceWriter& SourceWriter::Indent(int tab) {
  left_margin_.resize(
      std::max(static_cast<int>(left_margin_.size() + tab), 0), ' ');
  return *this;
}

ClassWriter* SourceWriter::BeginClass(const Type& clazz,
    const std::set<Type, Type::Comparator>& imports, int modifiers) {
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
  ClassWriter* class_writer = new ClassWriter(this);
  return class_writer->BeginClass(clazz, modifiers);
}

WriterInterface& operator<<(WriterInterface& writer, const Type& type) {
  WriteType(type, &writer);
  return writer;
}

WriterInterface& operator<<(WriterInterface& writer, const Snippet& snippet) {
  return writer.Write(snippet.data());
}

}  // namespace java
}  // namespace tensorflow
