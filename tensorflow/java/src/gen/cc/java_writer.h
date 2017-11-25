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
#include <set>
#include <vector>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/java/src/gen/cc/source_writer.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"

namespace tensorflow {
namespace java {

/// Path to the directory containing resource files for this generator
const char kGenResourcePath[] = "tensorflow/java/src/gen/resources/";

/// \brief Manipulator inserting a newline character.
inline void endl(SourceWriter* src_writer) {
  src_writer->EndOfLine();
}

/// \brief Manipulator beginning a new indented block of code.
inline void beginb(SourceWriter* src_writer) {
  src_writer->Write(src_writer->newline() ? "{" : " {")->EndOfLine()->Indent(2);
}

/// \brief Manipulator ending the current block of code.
inline void endb(SourceWriter* src_writer) {
  src_writer->Indent(-2)->Write("}")->EndOfLine();
}

/// \brief Basic streamer for outputting Java source code.
///
/// Specialized Java writers extends this class to expose common operators
/// for writing basic java code.
class JavaSourceStream {
 public:
  explicit JavaSourceStream(SourceWriter* src_writer)
      : src_writer_(src_writer) {}
  virtual ~JavaSourceStream() = default;

  /// \brief Applies the given manipulator method to this writer.
  JavaSourceStream& operator<<(void (*f)(SourceWriter*)) {
    f(src_writer_);
    return *this;
  }
  /// \brief Writes a piece of code or text.
  JavaSourceStream& operator<<(const string& str) {
    src_writer_->Write(str);
    return *this;
  }
  /// \brief Writes a piece of code or text.
  JavaSourceStream& operator<<(const char* str) {
    src_writer_->Write(str);
    return *this;
  }
  /// \brief Writes the signature of a type.
  JavaSourceStream& operator<<(const JavaType& type);

  /// \brief Writes a piece of code or text as read literally from a file.
  ///
  /// The snippet will be inlined at the current writing position, each line
  /// being indented properly.
  JavaSourceStream& operator<<(const JavaSnippet& snippet) {
    src_writer_->Inline(snippet.data());
    return *this;
  }

 protected:
  /// Underlying object to which we delegate the source code writing.
  SourceWriter* src_writer_;
};

/// \brief A utility for writing Java class methods.
///
/// This class can only be instantiated from a JavaClassWriter and should be
/// deleted implicitely by invoking EndOfMethod.
class JavaMethodWriter : public JavaSourceStream {
 public:
  /// \brief Ends the current method.
  ///
  /// This writer will become obsolete and be automatically discarded. No
  /// more call should be attempted on it thereafter.
  void EndMethod() {
    *this << endb;
    delete this;
  }

 private:
  std::set<string> declared_generics_;

  explicit JavaMethodWriter(SourceWriter* src_writer)
    : JavaSourceStream(src_writer) {}
  JavaMethodWriter(SourceWriter* src_writer, std::set<string> generics)
    : JavaSourceStream(src_writer), declared_generics_(generics) {}
  virtual ~JavaMethodWriter() = default;

  JavaMethodWriter* Begin(const JavaMethod& method, int modifiers);

  friend class JavaClassWriter;
};

/// \brief A utility for writing Java classes.
///
/// This class can only be instantiated from a JavaWriter or from another
/// JavaClassWriter when writing an inner class. It must be deleted implicitely
/// by invoking EndOfClass.
class JavaClassWriter : public JavaSourceStream {
 public:
  /// \brief Writes a list of variables as fields of this class.
  JavaClassWriter* WriteFields(const std::vector<JavaVar>& fields,
      int modifiers = 0);

  /// \brief Begins a method of this class.
  ///
  /// The returned writer should be used to write the content of the method and
  /// closed properly by calling EndOfMethod().
  JavaMethodWriter* BeginMethod(const JavaMethod& method, int modifiers = 0);

  /// \brief Begins a inner class of this class.
  ///
  /// The returned writer should be used to write the content of the inner class
  /// and closed properly by calling EndOfClass().
  JavaClassWriter* BeginInnerClass(const JavaType& clazz, int modifiers = 0);

  /// \brief Ends the current class.
  ///
  /// This writer will become obsolete and be automatically discarded. No
  /// more call should be attempted on it thereafter.
  void EndClass() {
    *this << endb;
    delete this;
  }

 private:
  std::set<string> declared_generics_;

  explicit JavaClassWriter(SourceWriter* src_writer)
    : JavaSourceStream(src_writer) {}
  JavaClassWriter(SourceWriter* src_writer, std::set<string> generics)
    : JavaSourceStream(src_writer), declared_generics_(generics) {}
  virtual ~JavaClassWriter() = default;

  JavaClassWriter* Begin(const JavaType& clazz, int modifiers);

  friend class JavaWriter;
};

/// \brief A utility for writing Java source code
///
/// It wraps a basic SourceWriter with an API specialized for writing Java
/// source code and based on definitions found in java_defs.h. The underlying
/// SourceWriter is not own by this object and should be released explicitly.
class JavaWriter : public JavaSourceStream {
 public:
  explicit JavaWriter(SourceWriter* src_writer)
      : JavaSourceStream(src_writer) {}
  virtual ~JavaWriter() = default;

  /// \brief Begins the main class.
  ///
  /// The returned writer should be used to write the content of the class and
  /// closed properly by calling EndOfClass().
  JavaClassWriter* BeginClass(const JavaType& clazz,
      const std::set<JavaType>& imports, int modifiers = 0);
};

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_WRITER_H_
