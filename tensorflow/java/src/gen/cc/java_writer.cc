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
#include "tensorflow/java/src/gen/cc/java_writer.h"

namespace tensorflow {

namespace {

void PrintModifiers(int modifiers, std::ostream& out) {
  if (modifiers & PUBLIC) {
    out << "public ";
  } else if (modifiers & PROTECTED) {
    out << "protected ";
  } else if (modifiers & PRIVATE) {
    out << "private ";
  }
  if (modifiers & STATIC) {
    out << "static ";
  }
  if (modifiers & FINAL) {
    out << "final ";
  }
}

void PrintType(const JavaType& type, std::ostream& out) {
  if (type.generic()) {
    out << (type.name().empty() ? "?" : type.name());
  } else {
    out << type.name();
    if (!type.params().empty()) {
      out << '<';
      std::list<JavaType>::const_iterator it;
      for (it = type.params().cbegin(); it != type.params().cend(); ++it) {
        if (it != type.params().cbegin()) {
          out << ", ";
        }
        PrintType(*it, out);
      }
      out << '>';
    }
  }
}

struct JavaTypePtrComparator {
  bool operator()(const JavaType* a, const JavaType* b) {
    return a->name().compare(b->name()) < 0;
  }
};

void PrintMethodGenerics(const JavaMethod& method, std::ostream& out,
    const std::set<string> declared_generics) {
  std::set<const JavaType*, JavaTypePtrComparator> generics;
  auto visitor = [&generics, &declared_generics](const JavaType* type) {
    if (type->generic() && !type->name().empty() &&
        declared_generics.find(type->name()) == declared_generics.end()) {
      generics.insert(type);
    }
  };
  method.type().Accept(visitor);
  for (std::list<JavaVariable>::const_iterator it = method.args().cbegin();
      it != method.args().cend(); ++it) {
    it->type().Accept(visitor);
  }
  if (!generics.empty()) {
    out << "<";
    for (std::set<const JavaType*>::const_iterator it = generics.cbegin();
        it != generics.cend(); ++it) {
      if (it != generics.cbegin()) {
        out << ", ";
      }
      PrintType(**it, out);
    }
    out << "> ";
  }
}

}  // namespace

void JavaBaseWriter::WriteAnnotations(
    const std::list<JavaAnnotation>& annotations) {
  std::list<JavaAnnotation>::const_iterator it;
  for (it = annotations.cbegin(); it != annotations.cend(); ++it) {
    stream()->Append("@" + it->name());
    if (!it->attrs().empty()) {
      stream()->Append("(" + it->attrs() + ")");
    }
    stream()->EndOfLine();
  }
}

void JavaBaseWriter::WriteDoc(const JavaDoc& doc,
    const std::list<JavaVariable>* params) {
  stream()->Append("/**")->EndOfLine()->Prefix(" * ");
  if (!doc.brief().empty()) {
    stream()->Inline(doc.brief())->EndOfLine();
  }
  if (!doc.description().empty()) {
    stream()->Append("<p>")->EndOfLine()->Inline(doc.description())->EndOfLine();
  }
  if (params != NULL) {
    stream()->EndOfLine();
    std::list<JavaVariable>::const_iterator it;
    for (it = params->begin(); it != params->end(); ++it) {
      stream()->Append("@param " + it->name() + " ")->Inline(it->doc().brief());
      stream()->EndOfLine();
    }
  }
  if (!doc.returnValue().empty()) {
    stream()->Inline("@return " + doc.returnValue())->EndOfLine();
  }
  stream()->RemovePrefix()->Append(" **/")->EndOfLine();
}

void JavaBaseWriter::WriteClassHeader(const JavaClass& clazz,
    int modifiers) {
  WriteDoc(clazz.doc());
  if (!clazz.annotations().empty()) {
    WriteAnnotations(clazz.annotations());
  }
  std::stringstream class_decl;
  PrintModifiers(modifiers, class_decl);
  class_decl << "class ";
  PrintType(clazz, class_decl);
  std::list<JavaType>::const_iterator it;
  for (it = clazz.params().cbegin(); it != clazz.params().cend(); ++it) {
    if (it->generic()) { // it should...
      context.declared_generics.insert(it->name());
    }
  }
  if (clazz.supertype() != nullptr) {
    class_decl << " extends ";
    PrintType(*clazz.supertype(), class_decl);
  }
  for (it = clazz.interfaces().cbegin(); it != clazz.interfaces().cend();
      ++it) {
    if (it == clazz.interfaces().cbegin()) {
      class_decl << " implements ";
    } else {
      class_decl << ", ";
    }
    PrintType(*it, class_decl);
  }
  BeginBlock(class_decl.str());
}

JavaClassWriter* JavaClassWriter::WriteFields(
    const std::list<JavaVariable>& fields, int modifiers) {
  stream()->EndOfLine();
  std::list<JavaVariable>::const_iterator it;
  for (it = fields.cbegin(); it != fields.cend(); ++it) {
    std::stringstream field_decl;
    PrintModifiers(modifiers, field_decl);
    PrintType(it->type(), field_decl);
    field_decl << " " << it->name() << ";";
    WriteLine(field_decl.str());
  }
  return this;
}

JavaMethodWriter* JavaClassWriter::BeginMethod(const JavaMethod& method,
    int modifiers) {
  stream()->EndOfLine();
  WriteDoc(method.doc(), &method.args());
  if (!method.annotations().empty()) {
    WriteAnnotations(method.annotations());
  }
  std::stringstream method_decl;
  PrintModifiers(modifiers, method_decl);
  PrintMethodGenerics(method, method_decl, (((modifiers & STATIC) == 0) ? context.declared_generics : std::set<string>()));
  if (!method.type().name().empty()) {
    PrintType(method.type(), method_decl);
    method_decl << " ";
  }
  method_decl << method.name() << "(";
  if (!method.args().empty()) {
    for (std::list<JavaVariable>::const_iterator it = method.args().cbegin();
        it != method.args().cend(); ++it) {
      if (it != method.args().cbegin()) {
        method_decl << ", ";
      }
      PrintType(it->type(), method_decl);
      method_decl << " " << it->name();
    }
  }
  method_decl << ")";
  BeginBlock(method_decl.str());
  return new JavaMethodWriter(context);
}

JavaClassWriter* JavaClassWriter::BeginInnerClass(const JavaClass& clazz,
    int modifiers) {
  stream()->EndOfLine();
  WriteClassHeader(clazz, modifiers);
  return new JavaClassWriter(context);
}

JavaClassWriter* JavaWriter::BeginClass(const JavaClass& clazz,
    const std::set<string>& imports, int modifiers) {
  WriteLine("package " + clazz.package() + ";");
  stream()->EndOfLine();
  std::set<string>::const_iterator it;
  for (it = imports.cbegin(); it != imports.cend(); ++it) {
    WriteLine("import " + *it + ";");
  }
  stream()->EndOfLine();
  WriteClassHeader(clazz, modifiers);
  return new JavaClassWriter(context);
}

}  // namespace tensorflow
