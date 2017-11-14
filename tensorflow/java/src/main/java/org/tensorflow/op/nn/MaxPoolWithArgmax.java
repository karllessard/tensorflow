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
import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class MaxPoolWithArgmax<T, U> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new MaxPoolWithArgmax operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param ksize
   * @param strides
   * @param Targmax
   * @param padding
   * @return a new instance of MaxPoolWithArgmax
   **/
  public static <T, U> MaxPoolWithArgmax<T, U> create(Scope scope, Operand<T> input, List<Integer> ksize, List<Integer> strides, Class<U> Targmax, String padding) {
    OperationBuilder opBuilder = scope.graph().opBuilder("MaxPoolWithArgmax", scope.makeOpName("MaxPoolWithArgmax"));
    opBuilder.addInput(input.asOutput());
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
    opBuilder.setAttr("Targmax", DataType.fromClass(Targmax));
    opBuilder.setAttr("padding", padding);
    return new MaxPoolWithArgmax<T, U>(opBuilder.build());
  }
  
  public Output<T> output() {
    return output;
  }
  
  public Output<U> argmax() {
    return argmax;
  }
  
  private Output<T> output;
  private Output<U> argmax;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private MaxPoolWithArgmax(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
    argmax = operation.output(outputIdx++);
  }
}
