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
package org.tensorflow.op.nn;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class QuantizedBiasAdd<V> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new QuantizedBiasAdd operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param bias
   * @param minInput
   * @param maxInput
   * @param minBias
   * @param maxBias
   * @param out_type
   * @return a new instance of QuantizedBiasAdd
   **/
  public static <T, U, V> QuantizedBiasAdd<V> create(Scope scope, Operand<T> input, Operand<U> bias, Operand<Float> minInput, Operand<Float> maxInput, Operand<Float> minBias, Operand<Float> maxBias, Class<V> out_type) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizedBiasAdd", scope.makeOpName("QuantizedBiasAdd"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(bias.asOutput());
    opBuilder.addInput(minInput.asOutput());
    opBuilder.addInput(maxInput.asOutput());
    opBuilder.addInput(minBias.asOutput());
    opBuilder.addInput(maxBias.asOutput());
    opBuilder.setAttr("out_type", DataType.fromClass(out_type));
    return new QuantizedBiasAdd<V>(opBuilder.build());
  }
  
  public Output<V> output() {
    return output;
  }
  
  public Output<Float> minOut() {
    return minOut;
  }
  
  public Output<Float> maxOut() {
    return maxOut;
  }
  
  private Output<V> output;
  private Output<Float> minOut;
  private Output<Float> maxOut;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private QuantizedBiasAdd(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
    minOut = operation.output(outputIdx++);
    maxOut = operation.output(outputIdx++);
  }
}
