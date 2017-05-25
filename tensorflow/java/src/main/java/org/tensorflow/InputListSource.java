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
 * Any object that could provide a list of tensors in input to an operator.
 *
 * <p>The tensors could be retrieved as an array of {@link Output} by invoking the {@link
 * InputListSource#inputs()} method.
 *
 * <p>Beware that even if it is somewhat confusing, an input tensor is always represented by an
 * instance of {@link Output}. This interface is only an abstraction of the object passing such
 * instance as an operand.
 */
public interface InputListSource {

  /** Returns an array of symbolic links of tensors to be passed in input. */
  Output[] inputs();
}
