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

#ifndef TENSORFLOW_JAVA_SRC_GEN_CC_SRC_OSTREAM_H_
#define TENSORFLOW_JAVA_SRC_GEN_CC_SRC_OSTREAM_H_

#include <memory>
#include <string>
#include <algorithm>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

class SourceOutputStream {
 public:
  SourceOutputStream() : new_line(true) {}
  virtual ~SourceOutputStream() {}

  SourceOutputStream* Append(const StringPiece& str);
  SourceOutputStream* Inline(const string& text);
  SourceOutputStream* EndOfLine() {
    static const StringPiece eol("\n");
    Append(eol);
    new_line = true;
    return this;
  }
  SourceOutputStream* Indent(int tab) {
    left_margin.resize(
        std::max(static_cast<int>(left_margin.size() + tab), 0), ' ');
    return this;
  }
  SourceOutputStream* Prefix(const char* line_prefix) {
    this->line_prefix = line_prefix;
    return this;
  }
  SourceOutputStream* RemovePrefix() {
    this->line_prefix.clear();
    return this;
  }

 protected:
  virtual void Output(const StringPiece& str) = 0;

 private:
  string left_margin;
  string line_prefix;
  bool new_line;
};

class FileSourceOutputStream : public SourceOutputStream {
 public:
  explicit FileSourceOutputStream(const string& fname,
      Env* env = Env::Default()) {
    TF_CHECK_OK(env->NewWritableFile(fname, &ofile));
  }
  virtual ~FileSourceOutputStream() {
    TF_CHECK_OK(ofile->Close());
  }
 protected:
  virtual void Output(const StringPiece& str) {
    TF_CHECK_OK(ofile->Append(str));
  }
 private:
  std::unique_ptr<WritableFile> ofile;
};

class StringSourceOutputStream : public SourceOutputStream {
 public:
  const string& ToString() {
    return buffer;
  }
 protected:
  virtual void Output(const StringPiece& str) {
    buffer.append(str.begin(), str.end());
  }
 private:
  string buffer;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_SRC_OSTREAM_H_
