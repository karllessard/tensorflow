package org.tensorflow.op.gen;

import static com.google.testing.compile.CompilationSubject.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import com.google.testing.compile.Compilation;
import com.google.testing.compile.Compiler;
import com.google.testing.compile.JavaFileObjects;

@RunWith(JUnit4.class)
public class OpGenCompileTest {

  @Test
  public void compileMultipleOutputsOp() {
    Compilation compile = compile("ops/test/MultipleOutputsOp.java");
    assertThat(compile).succeededWithoutWarnings();
  }

  @Test
  public void compileMultipleOutputsOpWithOptions() {
    Compilation compile = compile("ops/test/MultipleOutputsOpWithOptions.java");
    assertThat(compile).succeededWithoutWarnings();
  }

  @Test
  public void compileSingleOutputOp() {
    Compilation compile = compile("ops/test/SingleOutputOp.java");
    assertThat(compile).succeededWithoutWarnings();
  }

  @Test
  public void compileSingleOutputListOp() {
    Compilation compile = compile("ops/test/SingleOutputListOp.java");
    assertThat(compile).succeededWithoutWarnings();
  }

  @Test
  public void compileGenericOp() {
    Compilation compile = compile("ops/test/GenericOp.java");
    assertThat(compile).succeededWithoutWarnings();
  }

  private static Compilation compile(String path) {
    return Compiler.javac().compile(JavaFileObjects.forResource(path));
  }
}
