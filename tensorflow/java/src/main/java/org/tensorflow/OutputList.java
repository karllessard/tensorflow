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

/**
 * Holds a list of {@link Output} resulting from the output of an operator.
 *
 * <p>By implementing {@link InputListSource}, this class is convenient to pass an array of output
 * tensors in input to another operator.
 *
 * <pre>{@code
 * OutputList outputList = Ops.array().split(...).output();
 * ...
 * Ops.array().concat(outputList, ...);
 * }
 */
public class OutputList implements InputListSource {

  /**
   * Create an {@code OutputList} from the output of an operation.
   *
   * <p>The output must have a size > 1, where its first tensor is found at {@code op.output(start)}
   * and the last is {@code op.output(start + op.outputListLength(name) - 1)}.
   *
   * <p>This factory method collects at once all tensors in the list and stores them into an array.
   * This is preferred compared to an iterator since we do not want to repeat calls to the native
   * library if the list is visited more than once.
   *
   * @param op operation returning the output
   * @param start index of the first tensor of this output
   * @param name name of the output
   */
  public static OutputList create(Operation op, int start, String name) {
    int len = op.outputListLength(name);
    int end = start + len;

    Output[] array = new Output[len];
    for (int i = start; i < end; i++) {
      array[i - start] = op.output(i);
    }
    return new OutputList(array);
  }

  /**
   * Access the index-th {@link Output} in the list.
   *
   * @param index index of the {@link Output} in the list
   * @throws IndexOutOfBoundsException if index is out of list boundaries
   */
  public Output at(int index) {
    return array[index];
  }

  /** Returns the number of tensors in this list. */
  public int size() {
    return array.length;
  }

  /** Return the list of {@link Output} as an array. */
  public Output[] toArray() {
    return array;
  }

  @Override
  public Output[] inputs() {
    return toArray();
  }

  // Private constructor.
  private OutputList(Output[] array) {
    this.array = array;
  }

  private final Output[] array;
}
