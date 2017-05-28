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

import java.util.Iterator;

/**
 * Factory class of input types.
 *
 * <p>To be able to be passed as an input to an operation, the output of a previous operation should
 * implement one of the wrapper interfaces that extends from {@link Input} or {@link InputList}.
 *
 * <p>Since {@link Output} class implements directly the {@link Input} interface, no wrapping is
 * required for operation that returns simple output tensors.
 */
public final class Inputs {

  /**
   * Create an {@link InputList} out of the output of an operation.
   *
   * <p>The output must have a size > 1, where its first tensor is found at {@code op.output(start)}
   * and the last is {@code op.output(start + op.outputListLength(name) - 1)}.
   *
   * <p>This factory method fetches all tensors in the list and stores them into an array. This is
   * preferred compared to an iterator since we do not want to repeat calls to the native library if
   * the list is visited more than once.
   *
   * @param op operation to retrieve output from
   * @param start index of the first tensor of this output
   * @param length number of tensors in this output
   */
  public static InputList inputList(Operation op, int start, int length) {
    final Output[] outputs = fetchOutputs(op, start, length);

    return new InputList() {

      @Override
      public Output[] asOutputs() {
        return outputs;
      }

      @Override
      public Accessor<? extends Input> inputs() {

        return new Accessor<Input>() {

          @Override
          public Iterator<Input> iterator() {
            return new InputListIterator<Input>(this);
          }

          @Override
          public int size() {
            return outputs.length;
          }

          @Override
          public Input at(int index) {
            return outputs[index];
          }
        };
      }
    };
  }

  /**
   * Create a {@link VariableInput} out of the output of an operation.
   *
   * <p>This factory method fetches the tensor found at the provided index in the output list of the
   * operation and makes it available through {@link Input#asOutput()}.
   *
   * @param op operation to retrieve output from
   * @param index index of the output
   */
  public static VariableInput variableInput(Operation op, int index) {
    return variableInput(op.output(index));
  }

  /**
   * Create an {@link VariableInputList} out of the output of an operation.
   *
   * <p>This is identical to {@link #inputList} but it will return the list as an instance of {@link
   * VariableInputList} instead of a {@link InputList} to enforce compile-time type checking for
   * operations taking variable tensors in input.
   *
   * @param op operation to retrieve output from
   * @param start index of the first tensor of this output
   * @param length number of tensors in this output
   */
  public static InputList variableInputList(Operation op, int start, int length) {
    final Output[] outputs = fetchOutputs(op, start, length);

    return new VariableInputList() {

      @Override
      public Output[] asOutputs() {
        return outputs;
      }

      @Override
      public Accessor<? extends VariableInput> inputs() {

        return new Accessor<VariableInput>() {

          @Override
          public Iterator<VariableInput> iterator() {
            return new InputListIterator<VariableInput>(this);
          }

          @Override
          public int size() {
            return outputs.length;
          }

          @Override
          public VariableInput at(int index) {
            return Inputs.variableInput(outputs[index]);
          }
        };
      }
    };
  }

  // Create a {@link VariableInput} out of the output of an operation.
  // The output has already been fetched from the operation and instantiated as a {@link Output}.
  private static VariableInput variableInput(Output output) {

    return new VariableInput() {

      @Override
      public Output asOutput() {
        return output;
      }
    };
  }

  // All outputs of an operation are flattened in a sequence and retrieved individually using indexes. This
  // method will return the outputs found between 'start' and 'start + length - 1'.
  private static Output[] fetchOutputs(Operation op, int start, int length) {
    final Output[] outputs = new Output[length];
    int end = start + length;

    for (int i = start; i < end; i++) {
      outputs[i - start] = op.output(i);
    }
    return outputs;
  }

  // Basic implementation of an input list iterator.
  private static class InputListIterator<T extends Input> implements Iterator<T> {

    @Override
    public boolean hasNext() {
      return currentIndex < listAccessor.size();
    }

    @Override
    public T next() {
      return listAccessor.at(currentIndex++);
    }

    InputListIterator(InputList.Accessor<T> listAccessor) {
      this.listAccessor = listAccessor;
      currentIndex = 0;
    }

    private final InputList.Accessor<T> listAccessor;
    private int currentIndex;
  }

  private Inputs() {}
}
