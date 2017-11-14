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
package org.tensorflow.op.array;

import org.tensorflow.Operand;
import org.tensorflow.op.Operands;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class QuantizedConcat<T> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new QuantizedConcat operation to the graph.
   * 
   * @param scope Current graph scope
   * @param concatDim
   * @param values
   * @param inputMins
   * @param inputMaxes
   * @return a new instance of QuantizedConcat
   **/
  public static <T> QuantizedConcat<T> create(Scope scope, Operand<Integer> concatDim, Iterable<Operand<T>> values, Iterable<Operand<Float>> inputMins, Iterable<Operand<Float>> inputMaxes) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizedConcat", scope.makeOpName("QuantizedConcat"));
    opBuilder.addInput(concatDim.asOutput());
    opBuilder.addInputList(Operands.asOutputs(values));
    opBuilder.addInputList(Operands.asOutputs(inputMins));
    opBuilder.addInputList(Operands.asOutputs(inputMaxes));
    return new QuantizedConcat<T>(opBuilder.build());
  }
  
  public Output<T> output() {
    return output;
  }
  
  public Output<Float> outputMin() {
    return outputMin;
  }
  
  public Output<Float> outputMax() {
    return outputMax;
  }
  
  private Output<T> output;
  private Output<Float> outputMin;
  private Output<Float> outputMax;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private QuantizedConcat(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
    outputMin = operation.output(outputIdx++);
    outputMax = operation.output(outputIdx++);
  }
}
