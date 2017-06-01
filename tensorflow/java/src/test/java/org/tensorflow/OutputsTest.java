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

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.Outputs}. */
@RunWith(JUnit4.class)
public class OutputsTest {

  @Test
  public void createOutputList() {
    try (Graph g = new Graph()) {
      Operation op = TestUtil.split(g, new int[] {0, 1, 2}, 3);

      List<Output> outputs = Outputs.list(op, 0, 3); // get the right amount
      assertEquals(3, outputs.size());

      outputs = Outputs.list(op, 0, 1); // get less
      assertEquals(1, outputs.size());

      outputs = Outputs.list(op, 0, 4); // get more
      assertEquals(4, outputs.size());
    }
  }

  @Test
  public void createVariableOutputList() {
    try (Graph g = new Graph()) {
      Operation op = TestUtil.split(g, new int[] {0, 1, 2}, 3);

      List<VariableOutput> vars = Outputs.variableList(op, 0, 3); // get the right amount
      assertEquals(3, vars.size());

      vars = Outputs.variableList(op, 0, 1); // get less
      assertEquals(1, vars.size());

      vars = Outputs.variableList(op, 0, 4); // get more
      assertEquals(4, vars.size());
    }
  }

  @Test
  public void convertingInputsToOutputs() {
    try (Graph g = new Graph()) {
      Operation op = TestUtil.split(g, new int[] {0, 1, 2}, 3);

      List<Input> inputs = new ArrayList<>(3);
      for (int i = 0; i < 3; ++i) {
        inputs.add(op.output(i));
      }

      List<Output> outputs = Outputs.asOutputs(inputs);
      assertEquals(3, outputs.size());
    }
  }
}
