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

#include <vector>

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"
#include "tensorflow/java/src/gen/cc/java_writers.h"

namespace tensorflow {
namespace java {
namespace {

TEST(AppendTest, SingleLineText) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye and I say hello!");

  const char* expected = "You say goodbye and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(AppendTest, MultiLineText) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye\nand I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(AppendTest, MultiLineTextWithIndent) {
  SourceBufferWriter writer;
  writer.Indent(2).Append("You say goodbye\nand I say hello!");

  const char* expected = "  You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(AppendTest, MultiLineTextWithPrefix) {
  SourceBufferWriter writer;
  writer.SetLinePrefix("--").Append("You say goodbye\nand I say hello!");

  const char* expected = "--You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(AppendTest, MultiLineTextWithIndentAndPrefix) {
  SourceBufferWriter writer;
  writer.Indent(2).SetLinePrefix("--");
  writer.Append("You say goodbye\nand I say hello!");

  const char* expected = "  --You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteTest, SingleLineText) {
  SourceBufferWriter writer;
  writer.Write("You say goodbye and I say hello!");

  const char* expected = "You say goodbye and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteTest, MultiLineText) {
  SourceBufferWriter writer;
  writer.Write("You say goodbye\nand I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteTest, MultiLineTextWithIndent) {
  SourceBufferWriter writer;
  writer.Indent(2).Write("You say goodbye\nand I say hello!");

  const char* expected = "  You say goodbye\n  and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteTest, MultiLineTextWithPrefix) {
  SourceBufferWriter writer;
  writer.SetLinePrefix("--").Write("You say goodbye\nand I say hello!");

  const char* expected = "--You say goodbye\n--and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteTest, MultiLineTextWithIndentAndPrefix) {
  SourceBufferWriter writer;
  writer.Indent(2).SetLinePrefix("--");
  writer.Write("You say goodbye\nand I say hello!");

  const char* expected = "  --You say goodbye\n  --and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, Basic) {
  SourceBufferWriter writer;
  writer << "You say goodbye" << endl << "and I say hello!";

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, Indent) {
  SourceBufferWriter writer;
  writer << "You say goodbye" << endl;
  writer.Indent(2);
  writer << "and I say hello!";

  const char* expected = "You say goodbye\n  and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, IndentAndOutdent) {
  SourceBufferWriter writer;
  writer << "You say goodbye" << endl;
  writer.Indent(2);
  writer << "and I say hello!" << endl;
  writer.Indent(-2);
  writer << "Hello, hello!";

  const char* expected = "You say goodbye\n  and I say hello!\nHello, hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, SetLinePrefix) {
  SourceBufferWriter writer;
  writer << "You say goodbye" << endl;
  writer.SetLinePrefix("--");
  writer << "and I say hello!";

  const char* expected = "You say goodbye\n--and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, PrefixAndRemovePrefix) {
  SourceBufferWriter writer;
  writer << "You say goodbye" << endl;
  writer.SetLinePrefix("--");
  writer << "and I say hello!" << endl;
  writer.UnsetLinePrefix();
  writer << "Hello, hello!";

  const char* expected = "You say goodbye\n--and I say hello!\nHello, hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, IndentAndPrefixAndOutdentAndRemovePrefix) {
  SourceBufferWriter writer;
  writer << "You say goodbye" << endl;
  writer.Indent(2).SetLinePrefix("--");
  writer << "and I say hello!" << endl;
  writer.Indent(-2).UnsetLinePrefix();
  writer << "Hello, hello!";

  const char* expected = "You say goodbye\n  --and I say hello!\nHello, hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, NegativeIndent) {
  SourceBufferWriter writer;
  writer << "You say goodbye" << endl;
  writer.Indent(-10);
  writer << "and I say hello!";

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, CumulativeIndent) {
  SourceBufferWriter writer;
  writer << "You say goodbye" << endl;
  writer.Indent(2);
  writer << "and I say hello!" << endl;
  writer.Indent(2);
  writer << "Hello, hello!";

  const char* expected =
      "You say goodbye\n  and I say hello!\n    Hello, hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, EmptyPrefix) {
  SourceBufferWriter writer;
  writer << "You say goodbye" << endl;
  writer.SetLinePrefix("");
  writer << "and I say hello!";

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(StreamTest, BlocksAndLines) {
  SourceBufferWriter writer;

  writer << "int i = 0;" << endl
      << "int j = 10;" << endl
      << "if (true)"
      << beginb
      << "int aLongWayToTen = 0;" << endl
      << "while (++i <= j)"
      << beginb
      << "++aLongWayToTen;" << endl
      << endb
      << endb;

  const char* expected =
      "int i = 0;\n"
      "int j = 10;\n"
      "if (true) {\n"
      "  int aLongWayToTen = 0;\n"
      "  while (++i <= j) {\n"
      "    ++aLongWayToTen;\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(StreamTest, Types) {
  SourceBufferWriter writer;

  Type generic = Type::Generic("T").add_supertype(Type::Class("Number"));
  writer << Type::Int() << ", "
      << Type::Class("String") << ", "
      << generic << ", "
      << Type::ListOf(generic) << ", "
      << Type::ListOf(Type::IterableOf(generic)) << ", "
      << Type::ListOf(Type::Generic());

  const char* expected =
      "int, String, T, List<T>, List<Iterable<T>>, List<?>";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(StreamTest, Snippets) {
  SourceBufferWriter writer;

  Snippet snippet =
      Snippet::Create(io::JoinPath(kGenResourcePath, "test.snippet.java"));
  writer << snippet
      << beginb
      << snippet
      << endb;

  const char* expected =
      "// Here is a little snippet\n"
      "System.out.println(\"Hello!\");\n"
      "{\n"
      "  // Here is a little snippet\n"
      "  System.out.println(\"Hello!\");\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteClass, SimpleClass) {
  SourceBufferWriter writer;

  Type clazz = Type::Class("Test", "org.test");
  writer.BeginClass(clazz, Type::Set(), PUBLIC)->EndClass();

  const char* expected = "package org.test;\n\n"
      "public class Test {\n}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteClass, SimpleClassWithImports) {
  SourceBufferWriter writer;

  Type clazz = Type::Class("Test", "org.test");
  Type::Set imports;
  imports.insert(Type::Class("TypeA", "org.test.sub"));
  imports.insert(Type::Class("TypeA", "org.test.sub"));  // a second time
  imports.insert(Type::Class("TypeB", "org.other"));
  imports.insert(Type::Class("SamePackageType", "org.test"));
  imports.insert(Type::Class("NoPackageType"));
  writer.BeginClass(clazz, imports, PUBLIC)->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "import org.other.TypeB;\n"
      "import org.test.sub.TypeA;\n\n"
      "public class Test {\n}\n";
  ASSERT_STREQ(expected, writer.str().data());
}


TEST(WriteClass, AnnotatedAndDocumentedClass) {
  SourceBufferWriter writer;

  Type clazz = Type::Class("Test", "org.test");
  clazz.description("This class has a\n<p>\nmultiline description.");
  clazz.add_annotation(Annotation::Create("Bean"));
  clazz.add_annotation(Annotation::Create("SuppressWarnings")
      .attributes("\"rawtypes\""));
  writer.BeginClass(clazz, Type::Set(), PUBLIC)->EndClass();

  const char* expected = "package org.test;\n\n"
      "/**\n"
      " * This class has a\n"
      " * <p>\n"
      " * multiline description.\n"
      " **/\n"
      "@Bean\n"
      "@SuppressWarnings(\"rawtypes\")\n"
      "public class Test {\n}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteClass, ParameterizedClass) {
  SourceBufferWriter writer;

  Type clazz = Type::Class("Test", "org.test");
  clazz.add_parameter(Type::Generic("T"));
  clazz.add_parameter(Type::Generic("U").add_supertype(Type::Class("Number")));
  writer.BeginClass(clazz, Type::Set(), PUBLIC)->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test<T, U extends Number> {\n}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteClass, ParameterizedClassAndSupertypes) {
  SourceBufferWriter writer;

  Type clazz = Type::Class("Test", "org.test");
  Type type_t = Type::Generic("T");
  clazz.add_parameter(type_t);
  Type type_u = Type::Generic("U").add_supertype(Type::Class("Number"));
  clazz.add_parameter(type_u);
  clazz.add_supertype(Type::Class("SuperTest").add_parameter(type_t));
  clazz.add_supertype(Type::Interface("TestInf").add_parameter(type_u));
  writer.BeginClass(clazz, Type::Set(), PUBLIC)->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test<T, U extends Number>"
      " extends SuperTest<T> implements TestInf<U> {\n}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteClass, ParameterizedClassFields) {
  SourceBufferWriter writer;

  Type clazz = Type::Class("Test", "org.test");
  Type type_t = Type::Generic("T").add_supertype(Type::Class("Number"));
  clazz.add_parameter(type_t);
  ClassWriter* clazz_writer
      = writer.BeginClass(clazz, Type::Set(), PUBLIC);

  std::vector<Variable> static_fields;
  static_fields.push_back(Variable::Create("field1", Type::Class("String")));
  std::vector<Variable> member_fields;
  member_fields.push_back(Variable::Create("field2", Type::Class("String")));
  member_fields.push_back(Variable::Create("field3", type_t));

  clazz_writer->WriteFields(static_fields, STATIC | PUBLIC | FINAL)
      ->WriteFields(member_fields, PRIVATE)
      ->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test<T extends Number> {\n"
      "  \n"
      "  public static final String field1;\n"
      "  \n"
      "  private String field2;\n"
      "  private T field3;\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteClass, SimpleInnerClass) {
  SourceBufferWriter writer;

  Type clazz = Type::Class("Test", "org.test");
  ClassWriter* clazz_writer
      = writer.BeginClass(clazz, Type::Set(), PUBLIC);

  Type inner_class = Type::Class("InnerTest");
  clazz_writer->BeginInnerClass(inner_class, PUBLIC)->EndClass();
  clazz_writer->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test {\n"
      "  \n"
      "  public class InnerTest {\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteClass, StaticParameterizedInnerClass) {
  SourceBufferWriter writer;

  Type clazz = Type::Class("Test", "org.test");
  Type type_t = Type::Generic("T").add_supertype(Type::Class("Number"));
  clazz.add_parameter(type_t);
  ClassWriter* class_writer
      = writer.BeginClass(clazz, Type::Set(), PUBLIC);

  Type inner_class = Type::Class("InnerTest");
  inner_class.add_parameter(type_t);
  class_writer->BeginInnerClass(inner_class, PUBLIC | STATIC)->EndClass();
  class_writer->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test<T extends Number> {\n"
      "  \n"
      "  public static class InnerTest<T extends Number> {\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteMethod, SimpleMethod) {
  SourceBufferWriter writer;

  Type clazz = Type::Class("Test", "org.test");
  ClassWriter* clazz_writer
      = writer.BeginClass(clazz, Type::Set(), PUBLIC);

  Method method = Method::Create("doNothing", Type::Void());
  clazz_writer->BeginMethod(method, PUBLIC)->EndMethod();
  clazz_writer->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test {\n"
      "  \n"
      "  public void doNothing() {\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteMethod, AnnotatedAndDocumentedMethod) {
  SourceBufferWriter writer;

  Type clazz = Type::Class("Test", "org.test");
  ClassWriter* class_writer
      = writer.BeginClass(clazz, Type::Set(), PUBLIC);

  Method method = Method::Create("doNothing", Type::Void());
  method.description("This method has a\n<p>\nmultiline description.");
  method.add_annotation(Annotation::Create("Override"));
  method.add_annotation(Annotation::Create("SuppressWarnings")
      .attributes("\"rawtypes\""));
  class_writer->BeginMethod(method, PUBLIC)->EndMethod();
  class_writer->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test {\n"
      "  \n"
      "  /**\n"
      "   * This method has a\n"
      "   * <p>\n"
      "   * multiline description.\n"
      "   **/\n"
      "  @Override\n"
      "  @SuppressWarnings(\"rawtypes\")\n"
      "  public void doNothing() {\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteMethod, DocumentedMethodWithArguments) {
  SourceBufferWriter writer;

  Type clazz = Type::Class("Test", "org.test");
  ClassWriter* class_writer
      = writer.BeginClass(clazz, Type::Set(), PUBLIC);

  Method method = Method::Create("boolToInt", Type::Int());
  method.description("Converts a boolean to an int");
  method.return_description("int value for this boolean");
  method.add_argument(Variable::Create("b", Type::Boolean()));
  Variable reverse = Variable::Create("reverse", Type::Boolean());
  reverse.description("if true, value is reversed");
  method.add_argument(reverse);
  MethodWriter* method_writer = class_writer->BeginMethod(method, PUBLIC);
  *method_writer << "if (b && !reverse)"
      << beginb
      << "return 1;" << endl
      << endb
      << "return 0;" << endl;
  method_writer->EndMethod();
  class_writer->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test {\n"
      "  \n"
      "  /**\n"
      "   * Converts a boolean to an int\n"
      "   * \n"
      "   * @param b\n"
      "   * @param reverse if true, value is reversed\n"
      "   * @return int value for this boolean\n"
      "   **/\n"
      "  public int boolToInt(boolean b, boolean reverse) {\n"
      "    if (b && !reverse) {\n"
      "      return 1;\n"
      "    }\n"
      "    return 0;\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteMethod, ParameterizedMethod) {
  SourceBufferWriter writer;

  Type clazz = Type::Class("Test", "org.test");
  Type type_t = Type::Generic("T").add_supertype(Type::Class("Number"));
  clazz.add_parameter(type_t);
  ClassWriter* class_writer
      = writer.BeginClass(clazz, Type::Set(), PUBLIC);

  Method method = Method::Create("doNothing", type_t);
  MethodWriter* method_writer = class_writer->BeginMethod(method, PUBLIC);
  *method_writer << "return null;" << endl;
  method_writer->EndMethod();
  class_writer->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test<T extends Number> {\n"
      "  \n"
      "  public T doNothing() {\n"
      "    return null;\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteMethod, StaticParameterizedMethod) {
  SourceBufferWriter writer;

  Type clazz = Type::Class("Test", "org.test");
  Type type_t = Type::Generic("T").add_supertype(Type::Class("Number"));
  clazz.add_parameter(type_t);
  ClassWriter* clazz_writer
      = writer.BeginClass(clazz, Type::Set(), PUBLIC);

  Method method = Method::Create("doNothing", type_t);
  MethodWriter* method_writer =
      clazz_writer->BeginMethod(method, PUBLIC | STATIC);
  *method_writer << "return null;" << endl;
  method_writer->EndMethod();
  clazz_writer->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test<T extends Number> {\n"
      "  \n"
      "  public static <T extends Number> T doNothing() {\n"
      "    return null;\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

}  // namespace
}  // namespace java
}  // namespace tensorflow
