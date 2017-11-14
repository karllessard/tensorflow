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
package org.tensorflow.op.linalg;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class BatchMatrixSolve<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param adjoint
     **/
    public Options adjoint(Boolean adjoint) {
      this.adjoint = adjoint;
      return this;
    }
    
    private Boolean adjoint;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new BatchMatrixSolve operation to the graph.
   * 
   * @param scope Current graph scope
   * @param matrix
   * @param rhs
   * @return a new instance of BatchMatrixSolve
   **/
  public static <T> BatchMatrixSolve<T> create(Scope scope, Operand<T> matrix, Operand<T> rhs) {
    OperationBuilder opBuilder = scope.graph().opBuilder("BatchMatrixSolve", scope.makeOpName("BatchMatrixSolve"));
    opBuilder.addInput(matrix.asOutput());
    opBuilder.addInput(rhs.asOutput());
    return new BatchMatrixSolve<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new BatchMatrixSolve operation to the graph.
   * 
   * @param scope Current graph scope
   * @param matrix
   * @param rhs
   * @param options an object holding optional attributes values
   * @return a new instance of BatchMatrixSolve
   **/
  public static <T> BatchMatrixSolve<T> create(Scope scope, Operand<T> matrix, Operand<T> rhs, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("BatchMatrixSolve", scope.makeOpName("BatchMatrixSolve"));
    opBuilder.addInput(matrix.asOutput());
    opBuilder.addInput(rhs.asOutput());
    if (options.adjoint != null) {
      opBuilder.setAttr("adjoint", options.adjoint);
    }
    return new BatchMatrixSolve<T>(opBuilder.build());
  }
  
  /**
   * @param adjoint
   **/
  public static Options adjoint(Boolean adjoint) {
    return new Options().adjoint(adjoint);
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
  private BatchMatrixSolve(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
