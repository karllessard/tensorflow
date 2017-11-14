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
package org.tensorflow.op.sparse;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class SparseReduceSum<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param keepDims
     **/
    public Options keepDims(Boolean keepDims) {
      this.keepDims = keepDims;
      return this;
    }
    
    private Boolean keepDims;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new SparseReduceSum operation to the graph.
   * 
   * @param scope Current graph scope
   * @param inputIndices
   * @param inputValues
   * @param inputShape
   * @param reductionAxes
   * @return a new instance of SparseReduceSum
   **/
  public static <T> SparseReduceSum<T> create(Scope scope, Operand<Long> inputIndices, Operand<T> inputValues, Operand<Long> inputShape, Operand<Integer> reductionAxes) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseReduceSum", scope.makeOpName("SparseReduceSum"));
    opBuilder.addInput(inputIndices.asOutput());
    opBuilder.addInput(inputValues.asOutput());
    opBuilder.addInput(inputShape.asOutput());
    opBuilder.addInput(reductionAxes.asOutput());
    return new SparseReduceSum<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new SparseReduceSum operation to the graph.
   * 
   * @param scope Current graph scope
   * @param inputIndices
   * @param inputValues
   * @param inputShape
   * @param reductionAxes
   * @param options an object holding optional attributes values
   * @return a new instance of SparseReduceSum
   **/
  public static <T> SparseReduceSum<T> create(Scope scope, Operand<Long> inputIndices, Operand<T> inputValues, Operand<Long> inputShape, Operand<Integer> reductionAxes, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseReduceSum", scope.makeOpName("SparseReduceSum"));
    opBuilder.addInput(inputIndices.asOutput());
    opBuilder.addInput(inputValues.asOutput());
    opBuilder.addInput(inputShape.asOutput());
    opBuilder.addInput(reductionAxes.asOutput());
    if (options.keepDims != null) {
      opBuilder.setAttr("keepDims", options.keepDims);
    }
    return new SparseReduceSum<T>(opBuilder.build());
  }
  
  /**
   * @param keepDims
   **/
  public static Options keepDims(Boolean keepDims) {
    return new Options().keepDims(keepDims);
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
  private SparseReduceSum(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
