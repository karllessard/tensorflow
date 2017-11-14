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

public final class All extends PrimitiveOp implements Operand<Boolean> {
  
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
   * Factory method to create a class to wrap a new All operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param reductionIndices
   * @return a new instance of All
   **/
  public static <T> All create(Scope scope, Operand<Boolean> input, Operand<T> reductionIndices) {
    OperationBuilder opBuilder = scope.graph().opBuilder("All", scope.makeOpName("All"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(reductionIndices.asOutput());
    return new All(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new All operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param reductionIndices
   * @param options an object holding optional attributes values
   * @return a new instance of All
   **/
  public static <T> All create(Scope scope, Operand<Boolean> input, Operand<T> reductionIndices, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("All", scope.makeOpName("All"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(reductionIndices.asOutput());
    if (options.keepDims != null) {
      opBuilder.setAttr("keepDims", options.keepDims);
    }
    return new All(opBuilder.build());
  }
  
  /**
   * @param keepDims
   **/
  public static Options keepDims(Boolean keepDims) {
    return new Options().keepDims(keepDims);
  }
  
  public Output<Boolean> output() {
    return output;
  }
  
  @Override
  public Output<Boolean> asOutput() {
    return output;
  }
  
  private Output<Boolean> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private All(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
