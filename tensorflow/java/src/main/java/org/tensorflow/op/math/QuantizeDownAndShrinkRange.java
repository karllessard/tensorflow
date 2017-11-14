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
package org.tensorflow.op.math;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class QuantizeDownAndShrinkRange<U> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new QuantizeDownAndShrinkRange operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param inputMin
   * @param inputMax
   * @param out_type
   * @return a new instance of QuantizeDownAndShrinkRange
   **/
  public static <T, U> QuantizeDownAndShrinkRange<U> create(Scope scope, Operand<T> input, Operand<Float> inputMin, Operand<Float> inputMax, Class<U> out_type) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizeDownAndShrinkRange", scope.makeOpName("QuantizeDownAndShrinkRange"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(inputMin.asOutput());
    opBuilder.addInput(inputMax.asOutput());
    opBuilder.setAttr("out_type", DataType.fromClass(out_type));
    return new QuantizeDownAndShrinkRange<U>(opBuilder.build());
  }
  
  public Output<U> output() {
    return output;
  }
  
  public Output<Float> outputMin() {
    return outputMin;
  }
  
  public Output<Float> outputMax() {
    return outputMax;
  }
  
  private Output<U> output;
  private Output<Float> outputMin;
  private Output<Float> outputMax;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private QuantizeDownAndShrinkRange(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
    outputMin = operation.output(outputIdx++);
    outputMax = operation.output(outputIdx++);
  }
}
