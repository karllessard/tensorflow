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
package org.tensorflow.op.image;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class NonMaxSuppressionV2 extends PrimitiveOp implements Operand<Integer> {
  
  /**
   * Factory method to create a class to wrap a new NonMaxSuppressionV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param boxes
   * @param scores
   * @param maxOutputSize
   * @param iouThreshold
   * @return a new instance of NonMaxSuppressionV2
   **/
  public static NonMaxSuppressionV2 create(Scope scope, Operand<Float> boxes, Operand<Float> scores, Operand<Integer> maxOutputSize, Operand<Float> iouThreshold) {
    OperationBuilder opBuilder = scope.graph().opBuilder("NonMaxSuppressionV2", scope.makeOpName("NonMaxSuppressionV2"));
    opBuilder.addInput(boxes.asOutput());
    opBuilder.addInput(scores.asOutput());
    opBuilder.addInput(maxOutputSize.asOutput());
    opBuilder.addInput(iouThreshold.asOutput());
    return new NonMaxSuppressionV2(opBuilder.build());
  }
  
  public Output<Integer> selectedIndices() {
    return selectedIndices;
  }
  
  @Override
  public Output<Integer> asOutput() {
    return selectedIndices;
  }
  
  private Output<Integer> selectedIndices;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private NonMaxSuppressionV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    selectedIndices = operation.output(outputIdx++);
  }
}
