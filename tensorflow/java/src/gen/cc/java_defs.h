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

#include <memory>
#include <string>
#include <list>
#include <set>
#include <ostream>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/java/src/gen/cc/src_ostream.h"

namespace tensorflow {
namespace java {

enum Modifier {
  PUBLIC    = (1 << 0),
  PROTECTED = (1 << 1),
  PRIVATE   = (1 << 2),
  STATIC    = (1 << 3),
  FINAL     = (1 << 4),
};

class Doc {
 public:
  Doc() = default;
  virtual ~Doc() = default;

  Doc* brief(const string& brief) { brief_ = brief; return this; }
  const string& brief() const { return brief_; }
  Doc* description(const string& desc) { description_ = desc; return this; }
  const string& description() const { return description_; }
  Doc* returnValue(const string& ret) { return_value_ = ret; return this; }
  const string& returnValue() const { return return_value_; }

 private:
  string brief_;
  string description_;
  string return_value_;
};

class Type {
 public:
  explicit Type(bool generic = false) : generic_(generic) {}
  explicit Type(const string& name, bool generic = false)
    : generic_(generic), name_(name) {}
  Type(const string& name, const string& package)
    : name_(name), package_(package) {}
  explicit Type(const char* name)  // avoid clash with bool
    : Type(string(name)) {}
  Type(const string& name, const char* package)  // avoid clash with bool
    : Type(name, string(package)) {}
  virtual ~Type() = default;

  bool valid() const { return !name_.empty() || generic(); }
  bool generic() const { return generic_; }
  const string& name() const { return name_; }
  const string& package() const { return package_; }
  const Doc& doc() const { return doc_; }
  Type* doc(const Doc& doc) { doc_ = doc; return this; }
  Doc* doc_ptr() { return &doc_; }
  const std::list<Type>& params() const { return params_; }
  Type* param(const Type& param) {
    params_.push_back(param);
    return this;
  }
  Type* supertype(const Type& type) {
    supertype_ = std::shared_ptr<Type>(new Type(type));
    return this;
  }
  const Type* supertype_ptr() const { return supertype_.get(); }

  template <class TypeScanner> void Accept(TypeScanner* scanner) const;

 private:
  bool generic_ = false;
  string name_;
  string package_;
  std::list<Type> params_;
  std::shared_ptr<Type> supertype_;
  Doc doc_;
};

class Annotation : public Type {
 public:
  explicit Annotation(const string& name) : Type(name) {}
  Annotation(const string& name, const string& package)
    : Type(name, package) {}
  virtual ~Annotation() = default;

  const string& attrs() const { return attrs_; }
  Annotation* attrs(const string& attrs) { attrs_ = attrs; return this; }

 private:
  string attrs_;
};

class Class : public Type {
 public:
  Class() = default;
  explicit Class(const string& name) : Type(name) {}
  Class(const string& name, const string& package)
    : Type(name, package) {}
  virtual ~Class() = default;

  const std::list<Annotation>& annotations() const { return annotations_; }
  Class* annotation(const Annotation& annot) {
    annotations_.push_back(annot);
    return this;
  }
  const std::list<Type>& interfaces() const { return interfaces_; }
  Class* interface(const Type& type) {
    interfaces_.push_back(type);
    return this;
  }

  template <class TypeScanner> void Accept(TypeScanner* scanner) const;

 private:
  std::list<Annotation> annotations_;
  std::list<Type> interfaces_;
};

class Variable {
 public:
  Variable(const string& name, const Type& type)
    : name_(name), type_(type) {}
  virtual ~Variable() = default;

  const string& name() const { return name_; }
  const Type& type() const { return type_; }
  Type* type_ptr() { return &type_; }
  const Doc& doc() const { return doc_; }
  Variable* doc(const Doc& doc) { doc_ = doc; return this; }
  Doc* doc_ptr() { return &doc_; }

 private:
  string name_;
  Type type_;
  Doc doc_;
};

class Method {
 public:
  explicit Method(const string& name) : name_(name) {}
  Method(const string& name, const Type& type)
    : name_(name), type_(type) {}
  virtual ~Method() = default;

  const string& name() const { return name_; }
  const Type& type() const { return type_; }
  Type* type_ptr() { return &type_; }
  const Doc& doc() const { return doc_; }
  Method* doc(const Doc& doc) { doc_ = doc; return this; }
  Doc* doc_ptr() { return &doc_; }
  Method* args(const std::list<Variable>& args) {
    args_.insert(args_.cbegin(), args.cbegin(), args.cend());
    return this;
  }
  Method* arg(const Variable& var) {
    args_.push_back(var);
    return this;
  }
  const std::list<Variable>& args() const { return args_; }
  Method* annotation(const Annotation& annot) {
    annotations_.push_back(annot);
    return this;
  }
  const std::list<Annotation>& annotations() const { return annotations_; }

  template <class TypeScanner> void Accept(TypeScanner* scanner) const;

 private:
  string name_;
  Type type_;
  std::list<Variable> args_;
  std::list<Annotation> annotations_;
  Doc doc_;
};

template <class TypeScanner>
void Type::Accept(TypeScanner* scanner) const {
  (*scanner)(this);
  for (std::list<Type>::const_iterator it = params_.cbegin();
      it != params_.cend(); ++it) {
    it->Accept(scanner);
  }
  if (supertype_ptr() != nullptr) {
    supertype_ptr()->Accept(scanner);
  }
}

template <class TypeScanner>
void Class::Accept(TypeScanner* scanner) const {
  Type::Accept(scanner);
  for (std::list<Annotation>::const_iterator it = annotations_.cbegin();
      it != annotations_.cend(); ++it) {
    it->Accept(scanner);
  }
  for (std::list<Type>::const_iterator it = interfaces_.cbegin();
      it != interfaces_.cend(); ++it) {
    it->Accept(scanner);
  }
}

template <class TypeScanner>
void Method::Accept(TypeScanner* scanner) const {
  if (type_.valid()) {
    type_.Accept(scanner);
  }
  for (std::list<Variable>::const_iterator arg = args_.cbegin();
      arg != args_.cend(); ++arg) {
    arg->type().Accept(scanner);
  }
}

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_DEFS_H_
