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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Factory class of output-related types and lists. */
public final class Outputs {

  /**
   * Create a list of {@link Output} out of the output of an operation.
   *
   * <p>Since the {@link Output} class implements the {@link Input} interface, this list could be
   * passed directly in input to an operation.
   *
   * <pre>{@code
   * List<Output> outputs = ops.array().unpack(...).output();
   * ...
   * ops.array().pack(outputs);
   * }</pre>
   *
   * <p>The output must have a size >= 1, where its first tensor is found at {@code
   * op.output(start)} and the last is {@code op.output(start + op.outputListLength(name) - 1)}.
   *
   * <p>References to {@link Output} in the list are fetched at creation-time and stored into an
   * array to avoid repeated calls to the native library if the list is visited more than once.
   *
   * @param op operation to retrieve output from
   * @param start index of the first tensor of this output
   * @param length number of tensors in this output
   * @return a read-only list of {@link Output}
   */
  public static List<Output> list(Operation op, int start, int length) {
    List<Output> outputs = new ArrayList<>(length);
    int end = start + length;

    for (int i = start; i < end; i++) {
      outputs.add(op.output(i));
    }
    return Collections.unmodifiableList(outputs);
  }

  /**
   * Create a list of {@link VariableOutput} out of the output of an operation.
   *
   * <p>This is identical to {@link #list(Operation, int, int)} but outputs are returned as variable
   * handlers to enforce compile-time type checking for operations operating on variables.
   *
   * @param op operation to retrieve output from
   * @param start index of the first tensor of this output
   * @param length number of tensors in this output
   * @return a read-only list of {@link VariableOutput}
   * @see {@link #list(Operation, int, int)}
   */
  public static List<VariableOutput> variableList(Operation op, int start, int length) {
    List<VariableOutput> outputs = new ArrayList<>(length);
    int end = start + length;

    for (int i = start; i < end; i++) {
      outputs.add(op.variableOutput(i));
    }
    return Collections.unmodifiableList(outputs);
  }

  /**
   * Create a list of {@link Output} from a list of {@link Input}.
   *
   * <p>An {@link Input} is an interface wrapping an instance of {@link Output} allowing it to be
   * passed in input to an operation. The purpose of this method is to bring back any iteration of
   * {@link Input} to its {@link Output} nature.
   *
   * <pre>{@code
   * List<VariableInput> inputs = ...
   * List<Output> outputs = Inputs.outputList(inputs);
   * }</pre>
   *
   * @param inputs an iteration of inputs
   * @return a read-only list of {@link Output}
   */
  public static List<Output> asOutputs(Iterable<? extends Input> inputs) {
    List<Output> outputs = new ArrayList<>();

    for (Input input : inputs) {
      outputs.add(input.asOutput());
    }
    return Collections.unmodifiableList(outputs);
  }

  // Disabled constructor
  private Outputs() {}
}
