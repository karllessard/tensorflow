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

#ifndef TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_WRITER_H_
#define TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_WRITER_H_

#include <memory>
#include <string>
#include <list>
#include <set>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/java/src/gen/cc/src_ostream.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"

namespace tensorflow {
namespace java {

const char kGenResourcePath[] = "tensorflow/java/src/gen/resources/";

class BaseWriter {
 public:
  explicit BaseWriter(SourceOutputStream* stream) : stream_(stream) {}
  virtual ~BaseWriter() = default;

  BaseWriter* WriteSnippet(const string& fname, Env* env = Env::Default()) {
    string str;
    TF_CHECK_OK(ReadFileToString(env, fname, &str));
    stream_->Inline(str);
    return this;
  }
  BaseWriter* WriteLine(const string& str) {
    stream_->Append(str)->EndOfLine();
    return this;
  }
  BaseWriter* BeginBlock(const string& expr) {
    stream_->Append(expr);
    return BeginBlock();
  }
  BaseWriter* BeginBlock() {
    static const StringPiece open_brace(" {");
    stream_->Append(open_brace)->EndOfLine()->Indent(2);
    return this;
  }
  BaseWriter* EndOfBlock() {
    static const StringPiece close_brace("}");
    stream_->Indent(-2)->Append(close_brace)->EndOfLine();
    return this;
  }

 protected:
  SourceOutputStream* stream_;
};

class MethodWriter : public BaseWriter {
 public:
  explicit MethodWriter(SourceOutputStream * stream)
    : BaseWriter(stream) {}
  MethodWriter(SourceOutputStream* stream, std::set<string> generics)
    : BaseWriter(stream), declared_generics_names_(generics) {}
  virtual ~MethodWriter() = default;

  MethodWriter* WriteSnippet(const string& fname,
      Env* env = Env::Default()) {
    BaseWriter::WriteSnippet(fname, env);
    return this;
  }
  MethodWriter* WriteLine(const string& str) {
    BaseWriter::WriteLine(str);
    return this;
  }
  MethodWriter* BeginBlock(const string& expr) {
    BaseWriter::BeginBlock(expr);
    return this;
  }
  MethodWriter* EndOfBlock() {
    BaseWriter::EndOfBlock();
    return this;
  }
  void EndOfMethod() {
    BaseWriter::EndOfBlock();
    delete this;
  }

 private:
  std::set<string> declared_generics_names_;

  MethodWriter* Begin(const Method& method, int modifiers);

  friend class ClassWriter;
};

class ClassWriter : public BaseWriter {
 public:
  explicit ClassWriter(SourceOutputStream * stream)
    : BaseWriter(stream) {}
  ClassWriter(SourceOutputStream* stream, std::set<string> generics)
    : BaseWriter(stream), declared_generics_names_(generics) {}
  virtual ~ClassWriter() = default;

  ClassWriter* WriteFields(const std::list<Variable>& fields,
      int modifiers = 0);
  MethodWriter* BeginMethod(const Method& method, int modifiers = 0);
  ClassWriter* BeginInnerClass(const Class& clazz, int modifiers = 0);
  ClassWriter* WriteSnippet(const string& fname,
      Env* env = Env::Default()) {
    BaseWriter::WriteSnippet(fname, env);
    return this;
  }
  ClassWriter* WriteLine(const string& str) {
    BaseWriter::WriteLine(str);
    return this;
  }
  ClassWriter* BeginBlock(const string& expr) {
    BaseWriter::BeginBlock(expr);
    return this;
  }
  ClassWriter* EndOfBlock() {
    BaseWriter::EndOfBlock();
    return this;
  }
  void EndOfClass() {
    BaseWriter::EndOfBlock();
    delete this;
  }

 private:
  std::set<string> declared_generics_names_;

  ClassWriter* Begin(const Class& clazz, int modifiers);

  friend class Writer;
};

class Writer : public BaseWriter {
 public:
  explicit Writer(SourceOutputStream* stream) : BaseWriter(stream) {}
  virtual ~Writer() = default;

  ClassWriter* BeginClass(const Class& clazz,
      const std::set<string>& imports, int modifiers = 0);
  Writer* WriteSnippet(const string& fname, Env* env = Env::Default()) {
    BaseWriter::WriteSnippet(fname, env);
    return this;
  }
  Writer* WriteLine(const string& str) {
    BaseWriter::WriteLine(str);
    return this;
  }
};

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_WRITER_H_
