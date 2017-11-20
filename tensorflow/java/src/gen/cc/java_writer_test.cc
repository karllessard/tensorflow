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
#include "tensorflow/java/src/gen/cc/source_writer.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"
#include "tensorflow/java/src/gen/cc/java_writer.h"

namespace tensorflow {
namespace java {
namespace {

TEST(Basics, BlocksAndLines) {
  SourceBufferWriter writer;
  JavaSourceStream java_writer(&writer);

  java_writer << "int i = 0;" << endl
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
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(Basics, Types) {
  SourceBufferWriter writer;
  JavaSourceStream java_writer(&writer);

  JavaType generic = Java::Generic("T").supertype(Java::Class("Number"));
  java_writer << Java::Type("int") << ", "
      << Java::Class("String") << ", "
      << generic << ", "
      << Java::ListOf(generic) << ", "
      << Java::ListOf(Java::IterableOf(generic)) << ", "
      << Java::ListOf(Java::Wildcard());

  const char* expected =
      "int, String, T, List<T>, List<Iterable<T>>, List<?>";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(Basics, Snippets) {
  SourceBufferWriter writer;
  JavaSourceStream java_writer(&writer);

  JavaSnippet snippet(io::JoinPath(kGenResourcePath, "test.snippet.java"));
  java_writer << snippet
      << beginb
      << snippet
      << endb;

  const char* expected =
      "// I'm a little snippet\n"
      "// only used for a test\n"
      "{\n"
      "  // I'm a little snippet\n"
      "  // only used for a test\n"
      "}\n";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteClass, SimpleClass) {
  SourceBufferWriter writer;
  JavaWriter java_writer(&writer);

  JavaType clazz = Java::Class("Test", "org.test");
  java_writer.BeginClass(clazz, std::set<JavaType>(), PUBLIC)->EndClass();

  const char* expected = "package org.test;\n\n"
      "public class Test {\n}\n";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteClass, SimpleClassWithImports) {
  SourceBufferWriter writer;
  JavaWriter java_writer(&writer);

  JavaType clazz = Java::Class("Test", "org.test");
  std::set<JavaType> imports;
  imports.insert(Java::Class("TypeA", "org.test.sub"));
  imports.insert(Java::Class("TypeA", "org.test.sub"));  // a second time
  imports.insert(Java::Class("TypeB", "org.other"));
  imports.insert(Java::Class("SamePackageType", "org.test"));
  imports.insert(Java::Class("NoPackageType"));
  java_writer.BeginClass(clazz, imports, PUBLIC)->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "import org.test.sub.TypeA;\n"
      "import org.other.TypeB;\n\n"
      "public class Test {\n}\n";
  ASSERT_STREQ(expected, writer.ToString().data());
}


TEST(WriteClass, AnnotatedAndDocumentedClass) {
  SourceBufferWriter writer;
  JavaWriter java_writer(&writer);

  JavaType clazz = Java::Class("Test", "org.test");
  clazz.doc_ptr()->descr("This class has a\n<p>\nmultiline description.");
  clazz.annotation(Java::Annot("Bean"));
  clazz.annotation(Java::Annot("SuppressWarnings").attrs("\"rawtypes\""));
  java_writer.BeginClass(clazz, std::set<JavaType>(), PUBLIC)->EndClass();

  const char* expected = "package org.test;\n\n"
      "/**\n"
      " * This class has a\n"
      " * <p>\n"
      " * multiline description.\n"
      " **/\n"
      "@Bean\n"
      "@SuppressWarnings(\"rawtypes\")\n"
      "public class Test {\n}\n";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteClass, ParameterizedClass) {
  SourceBufferWriter writer;
  JavaWriter java_writer(&writer);

  JavaType clazz = Java::Class("Test", "org.test");
  clazz.param(Java::Generic("T"));
  clazz.param(Java::Generic("U").supertype(Java::Class("Number")));
  java_writer.BeginClass(clazz, std::set<JavaType>(), PUBLIC)->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test<T, U extends Number> {\n}\n";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteClass, ParameterizedClassAndSupertypes) {
  SourceBufferWriter writer;
  JavaWriter java_writer(&writer);

  JavaType clazz = Java::Class("Test", "org.test");
  JavaType type_t = Java::Generic("T");
  clazz.param(type_t);
  JavaType type_u = Java::Generic("U").supertype(Java::Class("Number"));
  clazz.param(type_u);
  clazz.supertype(Java::Class("SuperTest").param(type_t));
  clazz.supertype(Java::Interface("TestInf").param(type_u));
  java_writer.BeginClass(clazz, std::set<JavaType>(), PUBLIC)->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test<T, U extends Number>"
      " extends SuperTest<T> implements TestInf<U> {\n}\n";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteClass, ParameterizedClassFields) {
  SourceBufferWriter writer;
  JavaWriter java_writer(&writer);

  JavaType clazz = Java::Class("Test", "org.test");
  JavaType type_t = Java::Generic("T").supertype(Java::Class("Number"));
  clazz.param(type_t);
  JavaClassWriter* clazz_writer
      = java_writer.BeginClass(clazz, std::set<JavaType>(), PUBLIC);

  std::vector<JavaVar> static_fields;
  static_fields.push_back(Java::Var("field1", Java::Class("String")));
  std::vector<JavaVar> member_fields;
  member_fields.push_back(Java::Var("field2", Java::Class("String")));
  member_fields.push_back(Java::Var("field3", type_t));

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
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteClass, SimpleInnerClass) {
  SourceBufferWriter writer;
  JavaWriter java_writer(&writer);

  JavaType clazz = Java::Class("Test", "org.test");
  JavaClassWriter* clazz_writer
      = java_writer.BeginClass(clazz, std::set<JavaType>(), PUBLIC);

  JavaType inner_class = Java::Class("InnerTest");
  clazz_writer->BeginInnerClass(inner_class, PUBLIC)->EndClass();
  clazz_writer->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test {\n"
      "  \n"
      "  public class InnerTest {\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteClass, StaticParameterizedInnerClass) {
  SourceBufferWriter writer;
  JavaWriter java_writer(&writer);

  JavaType clazz = Java::Class("Test", "org.test");
  JavaType type_t = Java::Generic("T").supertype(Java::Class("Number"));
  clazz.param(type_t);
  JavaClassWriter* clazz_writer
      = java_writer.BeginClass(clazz, std::set<JavaType>(), PUBLIC);

  JavaType inner_class = Java::Class("InnerTest");
  inner_class.param(type_t);
  clazz_writer->BeginInnerClass(inner_class, PUBLIC | STATIC)->EndClass();
  clazz_writer->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test<T extends Number> {\n"
      "  \n"
      "  public static class InnerTest<T extends Number> {\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteMethod, SimpleMethod) {
  SourceBufferWriter writer;
  JavaWriter java_writer(&writer);

  JavaType clazz = Java::Class("Test", "org.test");
  JavaClassWriter* clazz_writer
      = java_writer.BeginClass(clazz, std::set<JavaType>(), PUBLIC);

  JavaMethod method = Java::Method("doNothing", Java::Type("void"));
  clazz_writer->BeginMethod(method, PUBLIC)->EndMethod();
  clazz_writer->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test {\n"
      "  \n"
      "  public void doNothing() {\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteMethod, AnnotatedAndDocumentedMethod) {
  SourceBufferWriter writer;
  JavaWriter java_writer(&writer);

  JavaType clazz = Java::Class("Test", "org.test");
  JavaClassWriter* clazz_writer
      = java_writer.BeginClass(clazz, std::set<JavaType>(), PUBLIC);

  JavaMethod method = Java::Method("doNothing", Java::Type("void"));
  method.doc_ptr()->descr("This method has a\n<p>\nmultiline description.");
  method.annotation(Java::Annot("Override"));
  method.annotation(Java::Annot("SuppressWarnings").attrs("\"rawtypes\""));
  clazz_writer->BeginMethod(method, PUBLIC)->EndMethod();
  clazz_writer->EndClass();

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
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteMethod, DocumentedMethodWithArguments) {
  SourceBufferWriter writer;
  JavaWriter java_writer(&writer);

  JavaType clazz = Java::Class("Test", "org.test");
  JavaClassWriter* clazz_writer
      = java_writer.BeginClass(clazz, std::set<JavaType>(), PUBLIC);

  JavaMethod method = Java::Method("boolToInt", Java::Type("int"));
  method.doc_ptr()->descr("Converts a boolean to an int");
  method.doc_ptr()->value("int value for this boolean");
  method.arg(Java::Var("b", Java::Type("boolean")));
  JavaVar reverse = Java::Var("reverse", Java::Type("boolean"));
  reverse.doc_ptr()->descr("if true, value is reversed");
  method.arg(reverse);
  JavaMethodWriter* method_writer = clazz_writer->BeginMethod(method, PUBLIC);
  (*method_writer) << "if (b && !reverse)"
      << beginb
      << "return 1;" << endl
      << endb
      << "return 0;" << endl;
  method_writer->EndMethod();
  clazz_writer->EndClass();

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
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteMethod, ParameterizedMethod) {
  SourceBufferWriter writer;
  JavaWriter java_writer(&writer);

  JavaType clazz = Java::Class("Test", "org.test");
  JavaType type_t = Java::Generic("T").supertype(Java::Class("Number"));
  clazz.param(type_t);
  JavaClassWriter* clazz_writer
      = java_writer.BeginClass(clazz, std::set<JavaType>(), PUBLIC);

  JavaMethod method = Java::Method("doNothing", type_t);
  JavaMethodWriter* method_writer = clazz_writer->BeginMethod(method, PUBLIC);
  (*method_writer) << "return null;" << endl;
  method_writer->EndMethod();
  clazz_writer->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test<T extends Number> {\n"
      "  \n"
      "  public T doNothing() {\n"
      "    return null;\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteMethod, StaticParameterizedMethod) {
  SourceBufferWriter writer;
  JavaWriter java_writer(&writer);

  JavaType clazz = Java::Class("Test", "org.test");
  JavaType type_t = Java::Generic("T").supertype(Java::Class("Number"));
  clazz.param(type_t);
  JavaClassWriter* clazz_writer
      = java_writer.BeginClass(clazz, std::set<JavaType>(), PUBLIC);

  JavaMethod method = Java::Method("doNothing", type_t);
  JavaMethodWriter* method_writer =
      clazz_writer->BeginMethod(method, PUBLIC | STATIC);
  (*method_writer) << "return null;" << endl;
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
  ASSERT_STREQ(expected, writer.ToString().data());
}

}  // namespace
}  // namespace java
}  // namespace tensorflow
