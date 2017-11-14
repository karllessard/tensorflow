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

public final class QuantizedRelu6<U> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new QuantizedRelu6 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param features
   * @param minFeatures
   * @param maxFeatures
   * @param out_type
   * @return a new instance of QuantizedRelu6
   **/
  public static <T, U> QuantizedRelu6<U> create(Scope scope, Operand<T> features, Operand<Float> minFeatures, Operand<Float> maxFeatures, Class<U> out_type) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizedRelu6", scope.makeOpName("QuantizedRelu6"));
    opBuilder.addInput(features.asOutput());
    opBuilder.addInput(minFeatures.asOutput());
    opBuilder.addInput(maxFeatures.asOutput());
    opBuilder.setAttr("out_type", DataType.fromClass(out_type));
    return new QuantizedRelu6<U>(opBuilder.build());
  }
  
  public Output<U> activations() {
    return activations;
  }
  
  public Output<Float> minActivations() {
    return minActivations;
  }
  
  public Output<Float> maxActivations() {
    return maxActivations;
  }
  
  private Output<U> activations;
  private Output<Float> minActivations;
  private Output<Float> maxActivations;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private QuantizedRelu6(Operation operation) {
    super(operation);
    int outputIdx = 0;
    activations = operation.output(outputIdx++);
    minActivations = operation.output(outputIdx++);
    maxActivations = operation.output(outputIdx++);
  }
}
