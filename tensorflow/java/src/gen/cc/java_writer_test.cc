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
  SourceStream java_writer(&writer);

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
  SourceStream java_writer(&writer);

  Type generic = Type::Generic("T").supertype(Type::Class("Number"));
  java_writer << Type::Type("int") << ", "
      << Type::Class("String") << ", "
      << generic << ", "
      << Type::ListOf(generic) << ", "
      << Type::ListOf(Type::IterableOf(generic)) << ", "
      << Type::ListOf(Type::Wildcard());

  const char* expected =
      "int, String, T, List<T>, List<Iterable<T>>, List<?>";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(Basics, Snippets) {
  SourceBufferWriter writer;
  SourceStream java_writer(&writer);

  Snippet snippet(io::JoinPath(kGenResourcePath, "test.snippet.java"));
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
  Writer java_writer(&writer);

  Type clazz = Type::Class("Test", "org.test");
  java_writer.BeginClass(clazz, std::set<Type>(), PUBLIC)->EndClass();

  const char* expected = "package org.test;\n\n"
      "public class Test {\n}\n";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteClass, SimpleClassWithImports) {
  SourceBufferWriter writer;
  Writer java_writer(&writer);

  Type clazz = Type::Class("Test", "org.test");
  std::set<Type> imports;
  imports.insert(Type::Class("TypeA", "org.test.sub"));
  imports.insert(Type::Class("TypeA", "org.test.sub"));  // a second time
  imports.insert(Type::Class("TypeB", "org.other"));
  imports.insert(Type::Class("SamePackageType", "org.test"));
  imports.insert(Type::Class("NoPackageType"));
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
  Writer java_writer(&writer);

  Type clazz = Type::Class("Test", "org.test");
  clazz.descr("This class has a\n<p>\nmultiline description.");
  clazz.annotation(Annotation::Of("Bean"));
  clazz.annotation(Annotation::Of("SuppressWarnings").attrs("\"rawtypes\""));
  java_writer.BeginClass(clazz, std::set<Type>(), PUBLIC)->EndClass();

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
  Writer java_writer(&writer);

  Type clazz = Type::Class("Test", "org.test");
  clazz.param(Type::Generic("T"));
  clazz.param(Type::Generic("U").supertype(Type::Class("Number")));
  java_writer.BeginClass(clazz, std::set<Type>(), PUBLIC)->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test<T, U extends Number> {\n}\n";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteClass, ParameterizedClassAndSupertypes) {
  SourceBufferWriter writer;
  Writer java_writer(&writer);

  Type clazz = Type::Class("Test", "org.test");
  Type type_t = Type::Generic("T");
  clazz.param(type_t);
  Type type_u = Type::Generic("U").supertype(Type::Class("Number"));
  clazz.param(type_u);
  clazz.supertype(Type::Class("SuperTest").param(type_t));
  clazz.supertype(Type::Interface("TestInf").param(type_u));
  java_writer.BeginClass(clazz, std::set<Type>(), PUBLIC)->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "public class Test<T, U extends Number>"
      " extends SuperTest<T> implements TestInf<U> {\n}\n";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteClass, ParameterizedClassFields) {
  SourceBufferWriter writer;
  Writer java_writer(&writer);

  Type clazz = Type::Class("Test", "org.test");
  Type type_t = Type::Generic("T").supertype(Type::Class("Number"));
  clazz.param(type_t);
  ClassWriter* clazz_writer
      = java_writer.BeginClass(clazz, std::set<Type>(), PUBLIC);

  std::vector<Variable> static_fields;
  static_fields.push_back(Variable::Of("field1", Type::Class("String")));
  std::vector<Variable> member_fields;
  member_fields.push_back(Variable::Of("field2", Type::Class("String")));
  member_fields.push_back(Variable::Of("field3", type_t));

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
  Writer java_writer(&writer);

  Type clazz = Type::Class("Test", "org.test");
  ClassWriter* clazz_writer
      = java_writer.BeginClass(clazz, std::set<Type>(), PUBLIC);

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
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteClass, StaticParameterizedInnerClass) {
  SourceBufferWriter writer;
  Writer java_writer(&writer);

  Type clazz = Type::Class("Test", "org.test");
  Type type_t = Type::Generic("T").supertype(Type::Class("Number"));
  clazz.param(type_t);
  ClassWriter* clazz_writer
      = java_writer.BeginClass(clazz, std::set<Type>(), PUBLIC);

  Type inner_class = Type::Class("InnerTest");
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
  Writer java_writer(&writer);

  Type clazz = Type::Class("Test", "org.test");
  ClassWriter* clazz_writer
      = java_writer.BeginClass(clazz, std::set<Type>(), PUBLIC);

  Method method = Method::Of("doNothing", Type::Type("void"));
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
  Writer java_writer(&writer);

  Type clazz = Type::Class("Test", "org.test");
  ClassWriter* clazz_writer
      = java_writer.BeginClass(clazz, std::set<Type>(), PUBLIC);

  Method method = Method::Of("doNothing", Type::Type("void"));
  method.descr("This method has a\n<p>\nmultiline description.");
  method.annotation(Annotation::Of("Override"));
  method.annotation(Annotation::Of("SuppressWarnings").attrs("\"rawtypes\""));
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
  Writer java_writer(&writer);

  Type clazz = Type::Class("Test", "org.test");
  ClassWriter* clazz_writer
      = java_writer.BeginClass(clazz, std::set<Type>(), PUBLIC);

  Method method = Method::Of("boolToInt", Type::Type("int"));
  method.descr("Converts a boolean to an int");
  method.doc_ptr()->value("int value for this boolean");
  method.arg(Variable::Of("b", Type::Type("boolean")));
  Variable reverse = Variable::Of("reverse", Type::Type("boolean"));
  reverse.descr("if true, value is reversed");
  method.arg(reverse);
  MethodWriter* method_writer = clazz_writer->BeginMethod(method, PUBLIC);
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
  Writer java_writer(&writer);

  Type clazz = Type::Class("Test", "org.test");
  Type type_t = Type::Generic("T").supertype(Type::Class("Number"));
  clazz.param(type_t);
  ClassWriter* clazz_writer
      = java_writer.BeginClass(clazz, std::set<Type>(), PUBLIC);

  Method method = Method::Of("doNothing", type_t);
  MethodWriter* method_writer = clazz_writer->BeginMethod(method, PUBLIC);
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
  Writer java_writer(&writer);

  Type clazz = Type::Class("Test", "org.test");
  Type type_t = Type::Generic("T").supertype(Type::Class("Number"));
  clazz.param(type_t);
  ClassWriter* clazz_writer
      = java_writer.BeginClass(clazz, std::set<Type>(), PUBLIC);

  Method method = Method::Of("doNothing", type_t);
  MethodWriter* method_writer =
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
