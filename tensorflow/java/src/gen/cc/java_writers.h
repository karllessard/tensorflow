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

#ifndef TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_WRITERS_H_
#define TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_WRITERS_H_

#include <memory>
#include <string>
#include <set>
#include <vector>
#include <algorithm>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"

namespace tensorflow {
namespace java {

// Common interface to all Java writers.
class WriterInterface {
 public:
  virtual ~WriterInterface() = default;

  // Appends a piece of code or text.
  //
  // It is expected that no newline character is present in the data provided,
  // otherwise Write() must be used.
  virtual WriterInterface& Append(const StringPiece& str) = 0;

  // Writes a block of code or text.
  //
  // The data might potentially contain newline characters, therefore it will
  // be scanned to ensure that each line is indented and prefixed properly,
  // making it a bit slower than Append().
  virtual WriterInterface& Write(const string& text) = 0;

  // Indents following lines with white spaces.
  //
  // Indentation is cumulative, i.e. the provided tabulation is added to the
  // current indentation value. If the tabulation is negative, the operation
  // will outdent the source code, until the indentation reaches 0 again.
  //
  // For example, calling Indent(2) twice will indent code with 4 white
  // spaces. Then calling Indent(-2) will outdent the code back to 2 white
  // spaces.
  virtual WriterInterface& Indent(int tab) = 0;

  // Prefixes following lines with provided character(s).
  //
  // A common use case of a prefix is for commenting or documenting the code.
  //
  // The prefix is written after the indentation, For example, invoking
  // Indent(2)->SetLinePrefix("//") will result in prefixing lines with "  //".
  virtual WriterInterface& SetLinePrefix(const char* line_prefix) = 0;

  // Removes the actual line prefix, if any.
  virtual WriterInterface& UnsetLinePrefix() = 0;

  // Appends a newline character and start writing on a new line.
  virtual WriterInterface& EndLine() = 0;

  // Returns true if the writer is at the beginnig of a new line
  virtual bool newline() const = 0;
};

// A subclass for writers that delegates writing operation to another writer.
//
// This is useful for sharing the context of a "root" writer with all other
// writers delegating their calls to it (e.g. margins, prefixes...).
class DelegatingWriter : public WriterInterface {
 public:
  explicit DelegatingWriter(WriterInterface* delegate)
      : delegate_(delegate) {}
  virtual ~DelegatingWriter() = default;

  WriterInterface& Append(const StringPiece& str) override  {
    return delegate_->Append(str);
  }
  WriterInterface& Write(const string& text) override  {
    return delegate_->Write(text);
  }
  WriterInterface& EndLine() override  {
    return delegate_->EndLine();
  }
  WriterInterface& Indent(int tab) override  {
    return delegate_->Indent(tab);
  }
  WriterInterface& SetLinePrefix(const char* line_prefix) override  {
    return delegate_->SetLinePrefix(line_prefix);
  }
  WriterInterface& UnsetLinePrefix() override  {
    return delegate_->UnsetLinePrefix();
  }
  bool newline() const override  {
    return delegate_->newline();
  }

 protected:
  WriterInterface* delegate_;
};

// A concrete class for writing Java methods.
//
// This class can only be instantiated from a ClassWriter and should be
// deleted implicitely by invoking EndMethod.
class MethodWriter : public DelegatingWriter {
 public:
  // Ends the current method.
  //
  // This writer will become obsolete and be automatically discarded. No
  // more call should be attempted on it thereafter.
  void EndMethod();

 private:
  friend class ClassWriter;

  explicit MethodWriter(WriterInterface* writer) : DelegatingWriter(writer) {}
  MethodWriter(WriterInterface* writer, std::set<string> generics)
    : DelegatingWriter(writer), declared_generics_(generics) {}
  virtual ~MethodWriter() = default;

  MethodWriter* BeginMethod(const Method& method, int modifiers);

  std::set<string> declared_generics_;
};

// A concrete class for writing Java classes.
//
// This class can only be instantiated from a Writer or from another
// ClassWriter when writing an inner class. It must be deleted implicitely
// by invoking EndClass.
class ClassWriter : public DelegatingWriter {
 public:
  // Writes a list of variables as fields of this class.
  ClassWriter* WriteFields(const std::vector<Variable>& fields,
      int modifiers = 0);

  // Begins a method of this class.
  //
  // The returned writer should be used to write the content of the method and
  // closed properly by calling EndMethod().
  MethodWriter* BeginMethod(const Method& method, int modifiers = 0);

  // Begins a inner class of this class.
  //
  // The returned writer should be used to write the content of the inner class
  // and closed properly by calling EndClass().
  ClassWriter* BeginInnerClass(const Type& clazz, int modifiers = 0);

  // Ends the current class.
  //
  // This writer will become obsolete and be automatically discarded. No
  // more call should be attempted on it thereafter.
  void EndClass();

 private:
  friend class SourceWriter;

  explicit ClassWriter(WriterInterface* writer)
    : DelegatingWriter(writer) {}
  ClassWriter(WriterInterface* writer, std::set<string> generics)
    : DelegatingWriter(writer), declared_generics_(generics) {}
  virtual ~ClassWriter() = default;

  ClassWriter* BeginClass(const Type& clazz, int modifiers);

  std::set<string> declared_generics_;
};

// A subclass for writing Java source code.
//
// Classes extending from this are at the root of the writer hierarchy and
// owns the context shared with other sub writers.
class SourceWriter : public WriterInterface {
 public:
  virtual ~SourceWriter() = default;

  SourceWriter& Append(const StringPiece& str) override;
  SourceWriter& Write(const string& text) override;
  SourceWriter& EndLine() override;
  SourceWriter& Indent(int tab) override;
  SourceWriter& SetLinePrefix(const char* line_prefix) override  {
    this->line_prefix_ = line_prefix;
    return *this;
  }
  SourceWriter& UnsetLinePrefix() override  {
    this->line_prefix_.clear();
    return *this;
  }
  bool newline() const override { return newline_; }

  // Begins the main class.
  //
  // The returned writer should be used to write the content of the class and
  // closed properly by calling EndClass().
  ClassWriter* BeginClass(const Type& clazz, const Type::Set& imports,
      int modifiers = 0);

 protected:
  virtual void DoAppend(const StringPiece& str) = 0;

 private:
  string left_margin_;
  string line_prefix_;
  bool newline_ = true;
};

// A concrete class for writing Java source code to a file.
//
// Note: the writer does not acquire the ownership of the file being passed in
// parameter.
class SourceFileWriter : public SourceWriter {
 public:
  explicit SourceFileWriter(WritableFile* file) : file_(file) {}
  virtual ~SourceFileWriter() = default;

 protected:
  void DoAppend(const StringPiece& str) override {
    TF_CHECK_OK(file_->Append(str));
  }
 private:
  WritableFile* file_;
};

// A concrete class for writing Java source code to a memory buffer.
class SourceBufferWriter : public SourceWriter {
 public:
  virtual ~SourceBufferWriter() = default;
  const string& str() {
    return buffer_;
  }
 protected:
  void DoAppend(const StringPiece& str) override {
    buffer_.append(str.begin(), str.end());
  }
 private:
  string buffer_;
};

//
// Stream interface
//

// Same as writer.Append(StringPiece).
inline WriterInterface& operator<<(WriterInterface& writer, const string& str) {
  return writer.Append(str);
}

// Same as writer.Append(StringPiece).
inline WriterInterface& operator<<(WriterInterface& writer, const char* str) {
  return writer.Append(str);
}

// Specialization for appending the signature of a Java type.
WriterInterface& operator<<(WriterInterface& writer, const Type& type);

// Specialization for writing the content of a Java snippet.
WriterInterface& operator<<(WriterInterface& writer, const Snippet& snippet);

// Invokes the given manipulator to this writer.
inline WriterInterface& operator<<(WriterInterface& writer,
    WriterInterface& (*f)(WriterInterface*)) {
  return f(&writer);
}

// Manipulator inserting a newline character.
inline WriterInterface& endl(WriterInterface* writer) {
  return writer->EndLine();
}

// Manipulator beginning a new indented block of code.
inline WriterInterface& beginb(WriterInterface* writer) {
  return writer->Append(writer->newline() ? "{" : " {").EndLine().Indent(2);
}

// Manipulator ending the current block of code.
inline WriterInterface& endb(WriterInterface* writer) {
  return writer->Indent(-2).Append("}").EndLine();
}

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_WRITERS_H_
