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

#include "tensorflow/core/platform/test.h"
#include "tensorflow/java/src/gen/cc/src_ostream.h"

namespace tensorflow {
namespace {

TEST(AppendTest, SingleLineText) {
  StringSourceOutputStream stream;
  stream.Append("You say goodbye and I say hello!");

  const char* expected = "You say goodbye and I say hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(AppendTest, MultiLineText) {
  StringSourceOutputStream stream;
  stream.Append("You say goodbye\nand I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(AppendTest, MultiLineTextWithIndent) {
  StringSourceOutputStream stream;
  stream.Indent(2)->Append("You say goodbye\nand I say hello!");

  const char* expected = "  You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(AppendTest, MultiLineTextWithPrefix) {
  StringSourceOutputStream stream;
  stream.Prefix("--")->Append("You say goodbye\nand I say hello!");

  const char* expected = "--You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(AppendTest, MultiLineTextWithIndentAndPrefix) {
  StringSourceOutputStream stream;
  stream.Indent(2)->Prefix("--")->Append("You say goodbye\nand I say hello!");

  const char* expected = "  --You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(InlineTest, SingleLineText) {
  StringSourceOutputStream stream;
  stream.Inline("You say goodbye and I say hello!");

  const char* expected = "You say goodbye and I say hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(InlineTest, MultiLineText) {
  StringSourceOutputStream stream;
  stream.Inline("You say goodbye\nand I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(InlineTest, MultiLineTextWithIndent) {
  StringSourceOutputStream stream;
  stream.Indent(2)->Inline("You say goodbye\nand I say hello!");

  const char* expected = "  You say goodbye\n  and I say hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(InlineTest, MultiLineTextWithPrefix) {
  StringSourceOutputStream stream;
  stream.Prefix("--")->Inline("You say goodbye\nand I say hello!");

  const char* expected = "--You say goodbye\n--and I say hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(InlineTest, MultiLineTextWithIndentAndPrefix) {
  StringSourceOutputStream stream;
  stream.Indent(2)->Prefix("--")->Inline("You say goodbye\nand I say hello!");

  const char* expected = "  --You say goodbye\n  --and I say hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(MarginTest, Basic) {
  StringSourceOutputStream stream;
  stream.Append("You say goodbye")
      ->EndOfLine()
      ->Append("and I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(MarginTest, Indent) {
  StringSourceOutputStream stream;
  stream.Append("You say goodbye")
      ->EndOfLine()
      ->Indent(2)
      ->Append("and I say hello!");

  const char* expected = "You say goodbye\n  and I say hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(MarginTest, IndentAndOutdent) {
  StringSourceOutputStream stream;
  stream.Append("You say goodbye")
      ->EndOfLine()
      ->Indent(2)
      ->Append("and I say hello!")
      ->EndOfLine()
      ->Indent(-2)
      ->Append("Hello, hello!");

  const char* expected = "You say goodbye\n  and I say hello!\nHello, hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(MarginTest, Prefix) {
  StringSourceOutputStream stream;
  stream.Append("You say goodbye")
      ->EndOfLine()
      ->Prefix("--")
      ->Append("and I say hello!");

  const char* expected = "You say goodbye\n--and I say hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(MarginTest, PrefixAndRemovePrefix) {
  StringSourceOutputStream stream;
  stream.Append("You say goodbye")
      ->EndOfLine()
      ->Prefix("--")
      ->Append("and I say hello!")
      ->EndOfLine()
      ->RemovePrefix()
      ->Append("Hello, hello!");

  const char* expected = "You say goodbye\n--and I say hello!\nHello, hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(MarginTest, IndentAndPrefixAndOutdentAndRemovePrefix) {
  StringSourceOutputStream stream;
  stream.Append("You say goodbye")
      ->EndOfLine()
      ->Indent(2)
      ->Prefix("--")
      ->Append("and I say hello!")
      ->EndOfLine()
      ->Indent(-2)
      ->RemovePrefix()
      ->Append("Hello, hello!");

  const char* expected = "You say goodbye\n  --and I say hello!\nHello, hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(MarginTest, NegativeIndent) {
  StringSourceOutputStream stream;
  stream.Append("You say goodbye")
      ->EndOfLine()
      ->Indent(-10)
      ->Append("and I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(MarginTest, CumulativeIndent) {
  StringSourceOutputStream stream;
  stream.Append("You say goodbye")
      ->EndOfLine()
      ->Indent(2)
      ->Append("and I say hello!")
      ->EndOfLine()
      ->Indent(2)
      ->Append("Hello, hello!");

  const char* expected =
      "You say goodbye\n  and I say hello!\n    Hello, hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}

TEST(MarginTest, EmptyPrefix) {
  StringSourceOutputStream stream;
  stream.Append("You say goodbye")
      ->EndOfLine()
      ->Prefix("")
      ->Append("and I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, stream.ToString().data());
}


}  // namespace
}  // namespace tensorflow
