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
 * Any object that could provide a tensor in input to an operation.
 *
 * <p>The tensor to be passed in input is retrieved as an {@link Output} by invoking the {@link
 * #asOutput()} method.
 *
 * <p>Beware that even if it is somewhat confusing, an input tensor is always represented by an
 * instance of {@link Output}. This interface is only an abstraction of the object passing the
 * tensor as an operand.
 */
public interface Input {

  /** Returns the input tensor as an {@link Output}. */
  Output asOutput();
}
