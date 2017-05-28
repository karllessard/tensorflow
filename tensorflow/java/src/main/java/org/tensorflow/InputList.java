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
 * Any object that could provide a list of tensors in input to an operation.
 *
 * <p>The tensors to be passed in input are retrieved as an array of {@link Output} by invoking the
 * {@link #asOutputs()} method.
 *
 * <p>Beware that even if it is somewhat confusing, an input tensor is always represented by an
 * instance of {@link Output}. This interface is only an abstraction of the object passing the list
 * of tensors as an operand.
 */
public interface InputList {

  /**
   * Entry point for accessing elements found in an {@code InputList}.
   *
   * <p>This is useful to access or iterate through the list of inputs without loosing their type,
   * as it would do by only accessing the array of outputs returned by {@link
   * InputList#asOutputs()}. For example:
   *
   * <pre>{@code
   * VariableInputList list = ...;
   * ops.training.applyGradientDescent(list.inputs().at(2), ...); // Takes a VariableInput in parameter
   * }</pre>
   *
   * or
   *
   * <pre>{@code
   * InputList list = ...;
   * for (Input input : list.inputs()) {
   *    ...
   * }
   * }</pre>
   */
  interface Accessor<T extends Input> extends Iterable<T> {

    /** Returns the number of inputs in the list. */
    int size();

    /**
     * Access the index-th input in the list.
     *
     * @param index index of the input in the list
     * @throws IndexOutOfBoundsException if index is out of list boundaries
     */
    T at(int index);
  }

  /** Returns the input tensors as an array of {@link Output}. */
  Output[] asOutputs();

  /** Allows access to the inputs of this list. */
  Accessor<? extends Input> inputs();
}
