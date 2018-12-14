/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.op;

import org.tensorflow.DataType;
import org.tensorflow.Output;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;

/**
 * A builder for {@link Operation}s.
 *
 * <p>For example, the following uses the builder to create an operation that produces 
 * the constant "3" as its output:
 *
 * <pre>{@code
 * // g is a Graph instance. // FIXME (karllessard) not instance of graph but instance of whatever can provide a OpBuilder
 * try (Tensor c1 = Tensor.create(3.0f)) {
 *   g.opBuilder("Const", "MyConst")
 *       .setAttr("dtype", c1.dataType())
 *       .setAttr("value", c1)
 *       .build();
 * }
 * }</pre>
 */
public interface NativeOperationBuilder {

  /**
   * Build the {@link Operation}.
   * 
   * <p>If this builder has been issued by a {@link Graph}, the operation will be added to the
   * graph to be executed later. If it has been issued by a {@link EagerSession}, it will be 
   * executed immediately.
   *
   * <p>The OperationBuilder is not usable after build() returns.
   */
  public NativeOperation build();

  /**
   * Add the output of another operation as the next input of the operation being built.
   *
   * @param input {@link Output} supposed to be the input of the operation being built.
   * @return the OperationBuilder instance for chaining.
   */
  public NativeOperationBuilder addInput(Output<?> input);

  /**
   * Add the outputs of another operation as the next inputs of the operation being built.
   *
   * @param inputs list of {@link Output} supposed to be the inputs of the operation being built.
   * @return the NativeOperationBuilder instance for chaining.
   */
  public NativeOperationBuilder addInputList(Output<?>[] inputs);

  /**
   * Set the device requested for computing the operation being built.
   * 
   * @param device the requested device, as a string
   * @return the NativeOperationBuilder instance for chaining.
   */
  public NativeOperationBuilder setDevice(String device);

  /**
   * Set the string values of an attribute of the operation being built. 
   * 
   * @param name attribute name
   * @param value attribute values
   * @return the NativeOperationBuilder instance for chaining.
   */
  public NativeOperationBuilder setAttr(String name, String[] value);

  /**
   * Set the string value of an attribute of the operation being built. 
   * 
   * @param name attribute name
   * @param value attribute value
   * @return the NativeOperationBuilder instance for chaining.
   */
  public NativeOperationBuilder setAttr(String name, String value);

  /**
   * Set the byte values of an attribute of the operation being built. 
   * 
   * @param name attribute name
   * @param value attribute values
   * @return the NativeOperationBuilder instance for chaining.
   */
  public NativeOperationBuilder setAttr(String name, byte[] value);

  /**
   * Set the long value of an attribute of the operation being built. 
   * 
   * @param name attribute name
   * @param value attribute value
   * @return the NativeOperationBuilder instance for chaining.
   */
  public NativeOperationBuilder setAttr(String name, long value);

  /**
   * Set the long values of an attribute of the operation being built. 
   * 
   * @param name attribute name
   * @param value attribute values
   * @return the NativeOperationBuilder instance for chaining.
   */
  public NativeOperationBuilder setAttr(String name, long[] value);

  /**
   * Set the float value of an attribute of the operation being built. 
   * 
   * @param name attribute name
   * @param value attribute value
   * @return the NativeOperationBuilder instance for chaining.
   */
  public NativeOperationBuilder setAttr(String name, float value);

  /**
   * Set the float values of an attribute of the operation being built. 
   * 
   * @param name attribute name
   * @param value attribute values
   * @return the NativeOperationBuilder instance for chaining.
   */
  public NativeOperationBuilder setAttr(String name, float[] value);

  /**
   * Set the boolean value of an attribute of the operation being built. 
   * 
   * @param name attribute name
   * @param value attribute value
   * @return the NativeOperationBuilder instance for chaining.
   */
  public NativeOperationBuilder setAttr(String name, boolean value);

  /**
   * Set the boolean values of an attribute of the operation being built. 
   * 
   * @param name attribute name
   * @param value attribute values
   * @return the NativeOperationBuilder instance for chaining.
   */
  public NativeOperationBuilder setAttr(String name, boolean[] value);

  /**
   * Set the type value of an attribute of the operation being built. 
   * 
   * @param name attribute name
   * @param value attribute value
   * @return the NativeOperationBuilder instance for chaining.
   */
  public NativeOperationBuilder setAttr(String name, DataType value);

  /**
   * Set the type values of an attribute of the operation being built. 
   * 
   * @param name attribute name
   * @param value attribute values
   * @return the NativeOperationBuilder instance for chaining.
   */
  public NativeOperationBuilder setAttr(String name, DataType[] value);

  /**
   * Set the tensor value of an attribute of the operation being built. 
   * 
   * @param name attribute name
   * @param value attribute value
   * @return the NativeOperationBuilder instance for chaining.
   */
  public NativeOperationBuilder setAttr(String name, Tensor<?> value);

  /**
   * Set the tensor values of an attribute of the operation being built. 
   * 
   * @param name attribute name
   * @param value attribute values
   * @return the NativeOperationBuilder instance for chaining.
   */
  public NativeOperationBuilder setAttr(String name, Tensor<?>[] value);

  /**
   * Set the shape value of an attribute of the operation being built. 
   * 
   * @param name attribute name
   * @param value attribute value
   * @return the NativeOperationBuilder instance for chaining.
   */
  public NativeOperationBuilder setAttr(String name, Shape value);

  /**
   * Set the shape values of an attribute of the operation being built. 
   * 
   * @param name attribute name
   * @param value attribute values
   * @return the NativeOperationBuilder instance for chaining.
   */
  public NativeOperationBuilder setAttr(String name, Shape[] value);
}
