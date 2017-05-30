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

package org.tensorflow;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.Inputs}. */
@RunWith(JUnit4.class)
public class InputsTest {

  @Test
  public void createOutputListFromOp() {
    try (Graph g = new Graph()) {
      Operation op = TestUtil.split(g, new int[] {0, 1, 2}, 3);

      List<Output> outputs = Inputs.outputList(op, 0, 3); // get the right amount
      assertEquals(3, outputs.size());

      outputs = Inputs.outputList(op, 0, 1); // get less
      assertEquals(1, outputs.size());

      outputs = Inputs.outputList(op, 0, 4); // get more
      assertEquals(4, outputs.size());
    }
  }

  @Test
  public void createOutputListFromInputs() {
    try (Graph g = new Graph()) {
      Operation op = TestUtil.split(g, new int[] {0, 1, 2}, 3);

      List<Input> inputs = new ArrayList<>(3);
      for (int i = 0; i < 3; ++i) {
        inputs.add(op.output(i));
      }

      List<Output> outputs = Inputs.outputList(inputs);
      assertEquals(3, outputs.size());
    }
  }

  @Test
  public void createVariableInputFromOp() {
    try (Graph g = new Graph()) {
      Operation op = TestUtil.split(g, new int[] {0, 1, 2}, 3);

      VariableInput var = Inputs.variableInput(op, 2);
      assertEquals(2, var.asOutput().index());

      var = Inputs.variableInput(op, 4); // out-of-bound
      assertEquals(4, var.asOutput().index());
    }
  }

  @Test
  public void createVariableInputFromOutput() {
    try (Graph g = new Graph()) {
      Output output = TestUtil.constant(g, "Const", new int[] {0});

      VariableInput var = Inputs.variableInput(output);
      assertSame(output, var.asOutput());
    }
  }

  @Test
  public void createVariableInputListFromOp() {
    try (Graph g = new Graph()) {
      Operation op = TestUtil.split(g, new int[] {0, 1, 2}, 3);

      List<VariableInput> vars = Inputs.variableInputList(op, 0, 3); // get the right amount
      assertEquals(3, vars.size());

      vars = Inputs.variableInputList(op, 0, 1); // get less
      assertEquals(1, vars.size());

      vars = Inputs.variableInputList(op, 0, 4); // get more
      assertEquals(4, vars.size());
    }
  }

  @Test
  public void createVariableInputListFromOutputs() {
    try (Graph g = new Graph()) {
      Operation op = TestUtil.split(g, new int[] {0, 1, 2}, 3);

      List<Output> outputs = new ArrayList<>(3);
      for (int i = 0; i < 3; ++i) {
        outputs.add(op.output(i));
      }

      List<VariableInput> vars = Inputs.variableInputList(outputs);
      assertEquals(3, vars.size());
    }
  }
}
