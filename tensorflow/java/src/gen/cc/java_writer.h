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

const char kJavaGenResourcePath[] = "tensorflow/java/src/gen/resources/";

class JavaBaseWriter {
 protected:
  struct Context {
    SourceOutputStream* stream;
    std::set<string> declared_generics;
  };

 public:
  explicit JavaBaseWriter(SourceOutputStream* stream) {
    context.stream = stream;
  }
  explicit JavaBaseWriter(Context context) : context(context) {}
  virtual ~JavaBaseWriter() = default;

  JavaBaseWriter* WriteSnippet(const string& fname, Env* env = Env::Default()) {
    string str;
    TF_CHECK_OK(ReadFileToString(env, fname, &str));
    stream()->Inline(str);
    return this;
  }
  JavaBaseWriter* WriteLine(const string& str) {
    stream()->Append(str)->EndOfLine();
    return this;
  }
  JavaBaseWriter* BeginBlock(const string& expr) {
    static const StringPiece open_brace(" {");
    stream()->Append(expr)->Append(open_brace)->EndOfLine()->Indent(2);
    return this;
  }
  JavaBaseWriter* EndOfBlock() {
    static const StringPiece close_brace("}");
    stream()->Indent(-2)->Append(close_brace)->EndOfLine();
    return this;
  }

 protected:
  Context context;
  SourceOutputStream* stream() { return context.stream; }

  void WriteAnnotations(const std::list<JavaAnnotation>& annotations);
  void WriteDoc(const JavaDoc& doc,
      const std::list<JavaVariable>* params = NULL);
  void WriteClassHeader(const JavaClass& clazz, int modifiers = 0);
};

class JavaMethodWriter : public JavaBaseWriter {
 public:
  explicit JavaMethodWriter(Context context)
    : JavaBaseWriter(context) {}
  virtual ~JavaMethodWriter() = default;

  JavaMethodWriter* WriteSnippet(const string& fname,
      Env* env = Env::Default()) {
    JavaBaseWriter::WriteSnippet(fname, env);
    return this;
  }
  JavaMethodWriter* WriteLine(const string& str) {
    JavaBaseWriter::WriteLine(str);
    return this;
  }
  JavaMethodWriter* BeginBlock(const string& expr) {
    JavaBaseWriter::BeginBlock(expr);
    return this;
  }
  JavaMethodWriter* EndOfBlock() {
    JavaBaseWriter::EndOfBlock();
    return this;
  }
  void EndOfMethod() {
    JavaBaseWriter::EndOfBlock();
    delete this;
  }
};

class JavaClassWriter : public JavaBaseWriter {
 public:
  explicit JavaClassWriter(Context context)
    : JavaBaseWriter(context) {}
  virtual ~JavaClassWriter() = default;

  JavaClassWriter* WriteFields(const std::list<JavaVariable>& fields,
      int modifiers = 0);
  JavaMethodWriter* BeginMethod(const JavaMethod& method, int modifiers = 0);
  JavaClassWriter* BeginInnerClass(const JavaClass& clazz, int modifiers = 0);
  JavaClassWriter* WriteSnippet(const string& fname,
      Env* env = Env::Default()) {
    JavaBaseWriter::WriteSnippet(fname, env);
    return this;
  }
  JavaClassWriter* WriteLine(const string& str) {
    JavaBaseWriter::WriteLine(str);
    return this;
  }
  JavaClassWriter* BeginBlock(const string& expr) {
    JavaBaseWriter::BeginBlock(expr);
    return this;
  }
  JavaClassWriter* EndOfBlock() {
    JavaBaseWriter::EndOfBlock();
    return this;
  }
  void EndOfClass() {
    JavaBaseWriter::EndOfBlock();
    delete this;
  }
};

class JavaWriter : public JavaBaseWriter {
 public:
  explicit JavaWriter(SourceOutputStream* stream) : JavaBaseWriter(stream) {}
  virtual ~JavaWriter() = default;

  JavaClassWriter* BeginClass(const JavaClass& clazz,
      const std::set<string>& imports, int modifiers = 0);
  JavaWriter* WriteSnippet(const string& fname, Env* env = Env::Default()) {
    JavaBaseWriter::WriteSnippet(fname, env);
    return this;
  }
  JavaWriter* WriteLine(const string& str) {
    JavaBaseWriter::WriteLine(str);
    return this;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_WRITER_H_
