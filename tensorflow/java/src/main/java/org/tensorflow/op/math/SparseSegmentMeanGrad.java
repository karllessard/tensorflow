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

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class SparseSegmentMeanGrad<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Factory method to create a class to wrap a new SparseSegmentMeanGrad operation to the graph.
   * 
   * @param scope Current graph scope
   * @param grad
   * @param indices
   * @param segmentIds
   * @param outputDim0
   * @return a new instance of SparseSegmentMeanGrad
   **/
  public static <T, U> SparseSegmentMeanGrad<T> create(Scope scope, Operand<T> grad, Operand<U> indices, Operand<Integer> segmentIds, Operand<Integer> outputDim0) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseSegmentMeanGrad", scope.makeOpName("SparseSegmentMeanGrad"));
    opBuilder.addInput(grad.asOutput());
    opBuilder.addInput(indices.asOutput());
    opBuilder.addInput(segmentIds.asOutput());
    opBuilder.addInput(outputDim0.asOutput());
    return new SparseSegmentMeanGrad<T>(opBuilder.build());
  }
  
  public Output<T> output() {
    return output;
  }
  
  @Override
  public Output<T> asOutput() {
    return output;
  }
  
  private Output<T> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private SparseSegmentMeanGrad(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
