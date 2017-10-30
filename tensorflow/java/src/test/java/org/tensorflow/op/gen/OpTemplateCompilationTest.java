package org.tensorflow.op.gen;

import static com.google.testing.compile.CompilationSubject.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import com.google.testing.compile.Compilation;
import com.google.testing.compile.Compiler;
import com.google.testing.compile.JavaFileObjects;

@RunWith(JUnit4.class)
public class OpTemplateCompilationTest {

  private static final Object[] JAVACOPTS = {}; // FIXME "-Xlint:all", "-source 7", "-target 7"};

  @Test
  public void compileSingleOutputOp() {
    Compilation compile = compile("test/SingleOutputOp.java");
    assertThat(compile).succeededWithoutWarnings();
  }

  @Test
  public void compileSingleOutputListOp() {
    Compilation compile = compile("test/SingleOutputListOp.java");
    assertThat(compile).succeededWithoutWarnings();
  }

  @Test
  public void compileMultipleAndMixedOutputsOp() {
    Compilation compile = compile("test/MultipleAndMixedOutputsOp.java");
    assertThat(compile).succeededWithoutWarnings();
  }

  @Test
  public void compileOptionalAndMandatoryAttributesOp() {
    Compilation compile = compile("test/OptionalAndMandatoryAttributesOp.java");
    assertThat(compile).succeededWithoutWarnings();
  }

  @Test
  public void compileStronglyTypedInputListOp() {
    Compilation compile = compile("test/StronglyTypedInputListOp.java");
    assertThat(compile).succeededWithoutWarnings();
  }

  @Test
  public void compileGenericWithParentOp() {
    Compilation compile = compile("test/GenericWithParentOp.java");
    assertThat(compile).succeededWithoutWarnings();
  }

  @Test
  public void compileGenericWithWildcardOp() {
    Compilation compile = compile("test/GenericWithWildcardOp.java");
    assertThat(compile).succeededWithoutWarnings();
  }

  @Test
  public void compileGenericWithExplicitCastOp() {
    Compilation compile = compile("test/GenericWithExplicitCastOp.java");
    assertThat(compile).succeededWithoutWarnings();
  }

  private static Compilation compile(String path) {
    return Compiler.javac().withOptions(JAVACOPTS).compile(JavaFileObjects.forResource(path));
  }
}
