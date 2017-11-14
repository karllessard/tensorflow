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

import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class QuantizedAvgPool<T> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new QuantizedAvgPool operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param minInput
   * @param maxInput
   * @param ksize
   * @param strides
   * @param padding
   * @return a new instance of QuantizedAvgPool
   **/
  public static <T> QuantizedAvgPool<T> create(Scope scope, Operand<T> input, Operand<Float> minInput, Operand<Float> maxInput, List<Integer> ksize, List<Integer> strides, String padding) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizedAvgPool", scope.makeOpName("QuantizedAvgPool"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(minInput.asOutput());
    opBuilder.addInput(maxInput.asOutput());
    long[] ksizeArray = new long[ksize.size()];
    for (int i = 0; i < ksizeArray.length; ++i) {
      ksizeArray[i] = ksize.get(i);
    }
    opBuilder.setAttr("ksize", ksizeArray);
    long[] stridesArray = new long[strides.size()];
    for (int i = 0; i < stridesArray.length; ++i) {
      stridesArray[i] = strides.get(i);
    }
    opBuilder.setAttr("strides", stridesArray);
    opBuilder.setAttr("padding", padding);
    return new QuantizedAvgPool<T>(opBuilder.build());
  }
  
  public Output<T> output() {
    return output;
  }
  
  public Output<Float> minOutput() {
    return minOutput;
  }
  
  public Output<Float> maxOutput() {
    return maxOutput;
  }
  
  private Output<T> output;
  private Output<Float> minOutput;
  private Output<Float> maxOutput;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private QuantizedAvgPool(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
    minOutput = operation.output(outputIdx++);
    maxOutput = operation.output(outputIdx++);
  }
}
