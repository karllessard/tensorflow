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

/// \brief Basic streamer for outputting  source code.
///
/// Specialized  writers extends this class to expose common operators
/// for writing basic java code.
class SourceStream {
 public:
  explicit SourceStream(SourceWriter* src_writer)
      : src_writer_(src_writer) {}
  virtual ~SourceStream() = default;

  /// \brief Applies the given manipulator method to this writer.
  SourceStream& operator<<(void (*f)(SourceWriter*)) {
    f(src_writer_);
    return *this;
  }
  /// \brief Writes a piece of code or text.
  SourceStream& operator<<(const string& str) {
    src_writer_->Write(str);
    return *this;
  }
  /// \brief Writes a piece of code or text.
  SourceStream& operator<<(const char* str) {
    src_writer_->Write(str);
    return *this;
  }
  /// \brief Writes the signature of a type.
  SourceStream& operator<<(const Type& type);

  /// \brief Writes a piece of code or text as read literally from a file.
  ///
  /// The snippet will be inlined at the current writing position, each line
  /// being indented properly.
  SourceStream& operator<<(const Snippet& snippet) {
    src_writer_->Inline(snippet.data());
    return *this;
  }

 protected:
  /// Underlying object to which we delegate the source code writing.
  SourceWriter* src_writer_;
};

/// \brief A utility for writing  class methods.
///
/// This class can only be instantiated from a ClassWriter and should be
/// deleted implicitely by invoking EndOfMethod.
class MethodWriter : public SourceStream {
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

  explicit MethodWriter(SourceWriter* src_writer)
    : SourceStream(src_writer) {}
  MethodWriter(SourceWriter* src_writer, std::set<string> generics)
    : SourceStream(src_writer), declared_generics_(generics) {}
  virtual ~MethodWriter() = default;

  MethodWriter* Begin(const Method& method, int modifiers);

  friend class ClassWriter;
};

/// \brief A utility for writing  classes.
///
/// This class can only be instantiated from a Writer or from another
/// ClassWriter when writing an inner class. It must be deleted implicitely
/// by invoking EndOfClass.
class ClassWriter : public SourceStream {
 public:
  /// \brief Writes a list of variables as fields of this class.
  ClassWriter* WriteFields(const std::vector<Variable>& fields,
      int modifiers = 0);

  /// \brief Begins a method of this class.
  ///
  /// The returned writer should be used to write the content of the method and
  /// closed properly by calling EndOfMethod().
  MethodWriter* BeginMethod(const Method& method, int modifiers = 0);

  /// \brief Begins a inner class of this class.
  ///
  /// The returned writer should be used to write the content of the inner class
  /// and closed properly by calling EndOfClass().
  ClassWriter* BeginInnerClass(const Type& clazz, int modifiers = 0);

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

  explicit ClassWriter(SourceWriter* src_writer)
    : SourceStream(src_writer) {}
  ClassWriter(SourceWriter* src_writer, std::set<string> generics)
    : SourceStream(src_writer), declared_generics_(generics) {}
  virtual ~ClassWriter() = default;

  ClassWriter* Begin(const Type& clazz, int modifiers);

  friend class Writer;
};

/// \brief A utility for writing  source code
///
/// It wraps a basic SourceWriter with an API specialized for writing
/// source code and based on definitions found in java_defs.h. The underlying
/// SourceWriter is not own by this object and should be released explicitly.
class Writer : public SourceStream {
 public:
  explicit Writer(SourceWriter* src_writer)
      : SourceStream(src_writer) {}
  virtual ~Writer() = default;

  /// \brief Begins the main class.
  ///
  /// The returned writer should be used to write the content of the class and
  /// closed properly by calling EndOfClass().
  ClassWriter* BeginClass(const Type& clazz,
      const std::set<Type>& imports, int modifiers = 0);
};

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_WRITER_H_
