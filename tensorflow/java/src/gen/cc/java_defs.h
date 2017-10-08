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

enum JavaModifier {
  PUBLIC    = (1 << 0),
  PROTECTED = (1 << 1),
  PRIVATE   = (1 << 2),
  STATIC    = (1 << 3),
  FINAL     = (1 << 4),
};

class JavaDoc {
 public:
  JavaDoc() = default;
  virtual ~JavaDoc() = default;
  JavaDoc* brief(const string& brief) { brief_ = brief; return this; }
  const string& brief() const { return brief_; }
  JavaDoc* description(const string& desc) { description_ = desc; return this; }
  const string& description() const { return description_; }
  JavaDoc* returnValue(const string& ret) { return_value_ = ret; return this; }
  const string& returnValue() const { return return_value_; }

 private:
  string brief_;
  string description_;
  string return_value_;
};


class JavaType {
 public:
  JavaType() = default;
  explicit JavaType(const string& name) : name_(name) {}
  JavaType(const string& name, const string& package)
    : name_(name), package_(package) {}
  virtual ~JavaType() = default;
  bool valid() const { return !name_.empty() || generic(); }
  const string& name() const { return name_; }
  JavaType* name(const string& name) { name_ = name; return this; }
  const string& package() const { return package_; }
  JavaType* package(const string& package) { package_ = package; return this; }
  bool generic() const { return generic_; }
  JavaType* generic(bool value) { generic_ = value; return this; }
  const JavaDoc& doc() const { return doc_; }
  JavaDoc* doc_ptr() { return &doc_; }
  JavaType* doc(const JavaDoc& doc) { doc_ = doc; return this; }
  const std::list<JavaType>& params() const { return params_; }
  JavaType* param(const JavaType& param) {
    params_.push_back(param);
    return this;
  }
  JavaType* supertype(const JavaType& type) {
    supertype_ = std::shared_ptr<JavaType>(new JavaType(type));
    return this;
  }
  const JavaType* supertype() const { return supertype_.get(); }

  template <class Visitor> void Accept(Visitor visitor) const;

 private:
  bool generic_ = false;
  JavaDoc doc_;
  string name_;
  string package_;
  std::list<JavaType> params_;
  std::shared_ptr<JavaType> supertype_;
};

class JavaAnnotation : public JavaType {
 public:
  JavaAnnotation() = default;
  explicit JavaAnnotation(const string& name) : JavaType(name) {}
  JavaAnnotation(const string& name, const string& package)
    : JavaType(name, package) {}
  virtual ~JavaAnnotation() = default;
  JavaAnnotation* attrs(const string& attrs) { attrs_ = attrs; return this; }
  const string& attrs() const { return attrs_; }

 private:
  string attrs_;
};

class JavaClass : public JavaType {
 public:
  JavaClass() = default;
  explicit JavaClass(const string& name) : JavaType(name) {}
  JavaClass(const string& name, const string& package)
    : JavaType(name, package) {}
  virtual ~JavaClass() = default;
  JavaClass* annotation(const JavaAnnotation& annot) {
    annotations_.push_back(annot);
    return this;
  }
  const std::list<JavaAnnotation>& annotations() const { return annotations_; }
  JavaClass* interface(const JavaType& type) {
    interfaces_.push_back(type);
    return this;
  }
  const std::list<JavaType>& interfaces() const { return interfaces_; }

  template <class Visitor> void Accept(Visitor visitor) const;

 private:
  std::list<JavaAnnotation> annotations_;
  std::list<JavaType> interfaces_;
};

class JavaVariable {
 public:
  JavaVariable() = default;
  JavaVariable(const string& name, const JavaType& type)
    : name_(name), type_(type) {}
  virtual ~JavaVariable() = default;
  JavaVariable* doc(const JavaDoc& doc) { doc_ = doc; return this; }
  JavaDoc* doc_ptr() { return &doc_; }
  const JavaDoc& doc() const { return doc_; }
  JavaVariable* name(const string& name) { name_ = name; return this; }
  const string& name() const { return name_; }
  JavaVariable* type(const JavaType& type) { type_ = type; return this; }
  JavaType* type_ptr() { return &type_; }
  const JavaType& type() const { return type_; }

 private:
  JavaDoc doc_;
  string name_;
  JavaType type_;
};

class JavaMethod {
 public:
  JavaMethod() = default;
  explicit JavaMethod(const string& name) : name_(name) {}
  JavaMethod(const string& name, const JavaType& type)
    : name_(name), type_(type) {}
  virtual ~JavaMethod() = default;
  JavaMethod* doc(const JavaDoc& doc) { doc_ = doc; return this; }
  JavaDoc* doc_ptr() { return &doc_; }
  const JavaDoc& doc() const { return doc_; }
  JavaMethod* name(const string& name) { name_ = name; return this; }
  const string& name() const { return name_; }
  JavaMethod* type(const JavaType& type) { type_ = type; return this; }
  JavaType* type_ptr() { return &type_; }
  const JavaType& type() const { return type_; }
  JavaMethod* args(const std::list<JavaVariable>& args) {
    args_.insert(args_.cbegin(), args.cbegin(), args.cend());
    return this;
  }
  JavaMethod* arg(const JavaVariable& var) {
    args_.push_back(var);
    return this;
  }
  const std::list<JavaVariable>& args() const { return args_; }
  JavaMethod* annotation(const JavaAnnotation& annot) {
    annotations_.push_back(annot);
    return this;
  }
  const std::list<JavaAnnotation>& annotations() const { return annotations_; }

 private:
  JavaDoc doc_;
  string name_;
  JavaType type_;
  std::list<JavaVariable> args_;
  std::list<JavaAnnotation> annotations_;
};

template <class Visitor>
void JavaType::Accept(Visitor visitor) const {
  visitor(this);
  for (std::list<JavaType>::const_iterator it = params_.cbegin();
      it != params_.cend(); ++it) {
    it->Accept(visitor);
  }
  if (supertype() != nullptr) {
    supertype()->Accept(visitor);
  }
}

template <class Visitor>
void JavaClass::Accept(Visitor visitor) const {
  JavaType::Accept(visitor);
  for (std::list<JavaAnnotation>::const_iterator it = annotations_.cbegin();
      it != annotations_.cend(); ++it) {
    it->Accept(visitor);
  }
  for (std::list<JavaType>::const_iterator it = interfaces_.cbegin();
      it != interfaces_.cend(); ++it) {
    it->Accept(visitor);
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_DEFS_H_
