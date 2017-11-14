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
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class MatrixBandPart<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Factory method to create a class to wrap a new MatrixBandPart operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param numLower
   * @param numUpper
   * @return a new instance of MatrixBandPart
   **/
  public static <T> MatrixBandPart<T> create(Scope scope, Operand<T> input, Operand<Long> numLower, Operand<Long> numUpper) {
    OperationBuilder opBuilder = scope.graph().opBuilder("MatrixBandPart", scope.makeOpName("MatrixBandPart"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(numLower.asOutput());
    opBuilder.addInput(numUpper.asOutput());
    return new MatrixBandPart<T>(opBuilder.build());
  }
  
  public Output<T> band() {
    return band;
  }
  
  @Override
  public Output<T> asOutput() {
    return band;
  }
  
  private Output<T> band;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private MatrixBandPart(Operation operation) {
    super(operation);
    int outputIdx = 0;
    band = operation.output(outputIdx++);
  }
}
