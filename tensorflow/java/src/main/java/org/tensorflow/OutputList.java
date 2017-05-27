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
 * Holds a list of {@link Output} resulting from the output of an operation.
 *
 * <p>By extending {@link InputList}, this interface is convenient to pass an array of output
 * tensors in input to another operation.
 *
 * <pre>{@code
 * OutputList outputList = Ops.array().split(...).output();
 * Output firstPiece = outputList.at(0);
 * ...
 * Ops.array().concat(outputList, ...);
 * }</pre>
 */
public interface OutputList extends InputList {

  /** Returns the number of tensors in this list. */
  int size();

  /**
   * Access the index-th {@link Output} in the list.
   *
   * @param index index of the {@link Output} in the list
   * @throws IndexOutOfBoundsException if index is out of list boundaries
   */
  Output at(int index);
}
