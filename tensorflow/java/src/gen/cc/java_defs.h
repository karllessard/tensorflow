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

#ifndef TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_DEFS_H_
#define TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_DEFS_H_

#include <string>
#include <vector>
#include <deque>

#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace java {

/// \brief An enumeration of different modifiers commonly used in Java
enum Modifier {
  PUBLIC    = (1 << 0),
  PROTECTED = (1 << 1),
  PRIVATE   = (1 << 2),
  STATIC    = (1 << 3),
  FINAL     = (1 << 4),
};

/// \brief A definition of a Java documentation block
///
/// Any vector of parameters (@param) that should be included in this block
/// can be provided separately (e.g. a vector of documented variables, see
/// JavaVariable).
class Doc {
 public:
  const string& descr() const { return descr_; }
  Doc& descr(const string& txt) { descr_ = txt; return *this; }
  const string& value() const { return value_; }
  Doc& value(const string& value) { value_ = value; return *this; }

 private:
  string descr_;
  string value_;
};

/// \brief A piece of code to read from a file.
class Snippet {
 public:
  explicit Snippet(const string& fname, Env* env = Env::Default()) {
    TF_CHECK_OK(ReadFileToString(env, fname, &data_));
  }
  const string& data() const { return data_; }

 private:
  string data_;
};

class Annotation;

/// \brief A definition of any kind of Java type (classes, interfaces...)
///
/// Note that most of the data fields of this class are only useful in specific
/// contexts and are not required in many cases. For example, annotations and
/// supertypes are only useful when declaring a type.
class Type {
 public:
  enum Kind {
    PRIMITIVE, CLASS, INTERFACE, ENUM, GENERIC, ANNOTATION
  };
  static Type Primitive(const string& name) {
    return Type(Type::PRIMITIVE, name, "");
  }
  static Type Class(const string& name, const string& package = "") {
    return Type(Type::CLASS, name, package);
  }
  static Type Interface(const string& name, const string& package = "") {
    return Type(Type::INTERFACE, name, package);
  }
  static Type Enum(const string& name, const string& package = "") {
    return Type(Type::ENUM, name, package);
  }
  static Type Generic(const string& name) {
    return Type(Type::GENERIC, name, "");
  }
  static Type Wildcard() {
    return Type(Type::GENERIC, "", "");
  }
  static Type ClassOf(const Type& type) {
    return Class("Class").param(type);
  }
  static Type ListOf(const Type& type) {
    return Interface("List", "java.util").param(type);
  }
  static Type IterableOf(const Type& type) {
    return Interface("Iterable").param(type);
  }
  const Kind& kind() const { return kind_; }
  const string& name() const { return name_; }
  const string& package() const { return package_; }
  const Doc& doc() const { return doc_; }
  Doc* mutable_doc() { return &doc_; }
  Type& doc(const Doc& doc) { doc_ = doc; return *this; }
  const std::vector<Type>& params() const { return params_; }
  Type& param(const Type& param) {
    params_.push_back(param);
    return *this;
  }
  const std::vector<Annotation>& annotations() const { return annotations_; }
  Type& annotation(const Annotation& annotation) {
    annotations_.push_back(annotation);
    return *this;
  }
  const std::deque<Type>& supertypes() const { return supertypes_; }
  Type& supertype(const Type& type) {
    if (type.kind_ == CLASS) {
      supertypes_.push_front(type);  // keep superclass at the front of the list
    } else if (type.kind_ == INTERFACE) {
      supertypes_.push_back(type);
    }
    return *this;
  }
  /// Returns true if "type" is of a known collection type (only a few for now)
  bool IsCollection() {
    return name_ == "List" || name_ == "Iterable";
  }
  /// Scans this type and any of its parameter types.
  template <class TypeScanner> void Scan(TypeScanner* scanner) const;

 protected:
  Type(Kind kind, const string& name, const string& package)
    : kind_(kind), name_(name), package_(package) {}

 private:
  Kind kind_;
  string name_;
  string package_;
  std::vector<Type> params_;
  std::vector<Annotation> annotations_;
  std::deque<Type> supertypes_;
  Doc doc_;
};

/// \brief Definition of a Java annotation
///
/// This class only defines the usage of an annotation in a specific context,
/// giving optionally a set of attributes to initialize.
class Annotation : public Type {
 public:
  static Annotation OfType(const string& type_name, const string& pkg = "") {
    return Type(Type::ANNOTATION, type_name, pkg);
  }
  const string& attrs() const { return attrs_; }
  Annotation& attrs(const string& attrs) { attrs_ = attrs; return *this; }

 private:
  string attrs_;
};

/// \brief A definition of a Java variable
///
/// This class defines an instance of a type, which could be documented.
class Variable {
 public:
  static Variable Field(const string& name, const Type& type) {
    return Variable(name, type, false);
  }
  static Variable Arg(const string& name, const Type& type) {
    return Variable(name, type, false);
  }
  static Variable VarArg(const string& name, const Type& type) {
    return Variable(name, type, true);
  }
  const string& name() const { return name_; }
  const Type& type() const { return type_; }
  bool variadic() const { return variadic_; }
  const Doc& doc() const { return doc_; }
  Doc* mutable_doc() { return &doc_; }
  Variable& doc(const Doc& doc) { doc_ = doc; return *this; }

 private:
  string name_;
  Type type_;
  bool variadic_;
  Doc doc_;

  Variable(const string& name, const Type& type, bool variadic)
    : name_(name), type_(type), variadic_(variadic) {}
};

/// \brief A definition of a Java class method
///
/// This class defines the signature of a method, including its name, return
/// type and arguments.
class Method {
 public:
  static Method ConstructorFor(const Type& clazz) {
    return Method(clazz.name(), clazz, true);
  }
  static Method Member(const string& name, const Type& return_type) {
    return Method(name, return_type, false);
  }
  const string& name() const { return name_; }
  const Type& return_type() const { return return_type_; }
  bool constructor() const { return constructor_; }
  const Doc& doc() const { return doc_; }
  Doc* mutable_doc() { return &doc_; }
  Method& doc(const Doc& doc) { doc_ = doc; return *this; }
  const std::vector<Variable>& args() const { return args_; }
  Method& args(const std::vector<Variable>& args) {
    args_.insert(args_.cend(), args.cbegin(), args.cend());
    return *this;
  }
  Method& arg(const Variable& var) { args_.push_back(var); return *this; }
  const std::vector<Annotation>& annotations() const { return annotations_; }
  Method& annotation(const Annotation& annotation) {
    annotations_.push_back(annotation);
    return *this;
  }
  /// Scans all types found in the signature of this method.
  template <class TypeScanner>
  void ScanTypes(TypeScanner* scanner, bool args_only) const;

 private:
  string name_;
  Type return_type_;
  bool constructor_;
  std::vector<Variable> args_;
  std::vector<Annotation> annotations_;
  Doc doc_;

  Method(const string& name, const Type& return_type, bool constructor)
    : name_(name), return_type_(return_type), constructor_(constructor) {}
};

// Templates implementation

template <class TypeScanner>
void Type::Scan(TypeScanner* scanner) const {
  (*scanner)(this);
  for (std::vector<Type>::const_iterator it = params_.cbegin();
      it != params_.cend(); ++it) {
    it->Scan(scanner);
  }
  for (std::vector<Annotation>::const_iterator it = annotations_.cbegin();
      it != annotations_.cend(); ++it) {
    it->type().Scan(scanner);
  }
  for (std::deque<Type>::const_iterator it = supertypes_.cbegin();
      it != supertypes_.cend(); ++it) {
    it->Scan(scanner);
  }
}

template <class TypeScanner>
void Method::ScanTypes(TypeScanner* scanner, bool args_only) const {
  if (!args_only && !constructor()) {
    return_type_.Scan(scanner);
  }
  for (std::vector<Variable>::const_iterator arg = args_.cbegin();
      arg != args_.cend(); ++arg) {
    arg->type().Scan(scanner);
  }
}

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_DEFS_H_
