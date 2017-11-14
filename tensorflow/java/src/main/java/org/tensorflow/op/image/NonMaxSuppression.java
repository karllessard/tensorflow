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

public final class NonMaxSuppression extends PrimitiveOp implements Operand<Integer> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param iouThreshold
     **/
    public Options iouThreshold(Float iouThreshold) {
      this.iouThreshold = iouThreshold;
      return this;
    }
    
    private Float iouThreshold;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new NonMaxSuppression operation to the graph.
   * 
   * @param scope Current graph scope
   * @param boxes
   * @param scores
   * @param maxOutputSize
   * @return a new instance of NonMaxSuppression
   **/
  public static NonMaxSuppression create(Scope scope, Operand<Float> boxes, Operand<Float> scores, Operand<Integer> maxOutputSize) {
    OperationBuilder opBuilder = scope.graph().opBuilder("NonMaxSuppression", scope.makeOpName("NonMaxSuppression"));
    opBuilder.addInput(boxes.asOutput());
    opBuilder.addInput(scores.asOutput());
    opBuilder.addInput(maxOutputSize.asOutput());
    return new NonMaxSuppression(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new NonMaxSuppression operation to the graph.
   * 
   * @param scope Current graph scope
   * @param boxes
   * @param scores
   * @param maxOutputSize
   * @param options an object holding optional attributes values
   * @return a new instance of NonMaxSuppression
   **/
  public static NonMaxSuppression create(Scope scope, Operand<Float> boxes, Operand<Float> scores, Operand<Integer> maxOutputSize, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("NonMaxSuppression", scope.makeOpName("NonMaxSuppression"));
    opBuilder.addInput(boxes.asOutput());
    opBuilder.addInput(scores.asOutput());
    opBuilder.addInput(maxOutputSize.asOutput());
    if (options.iouThreshold != null) {
      opBuilder.setAttr("iouThreshold", options.iouThreshold);
    }
    return new NonMaxSuppression(opBuilder.build());
  }
  
  /**
   * @param iouThreshold
   **/
  public static Options iouThreshold(Float iouThreshold) {
    return new Options().iouThreshold(iouThreshold);
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
  private NonMaxSuppression(Operation operation) {
    super(operation);
    int outputIdx = 0;
    selectedIndices = operation.output(outputIdx++);
  }
}
