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

public final class MatrixSolveLs<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param fast
     **/
    public Options fast(Boolean fast) {
      this.fast = fast;
      return this;
    }
    
    private Boolean fast;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new MatrixSolveLs operation to the graph.
   * 
   * @param scope Current graph scope
   * @param matrix
   * @param rhs
   * @param l2Regularizer
   * @return a new instance of MatrixSolveLs
   **/
  public static <T> MatrixSolveLs<T> create(Scope scope, Operand<T> matrix, Operand<T> rhs, Operand<Double> l2Regularizer) {
    OperationBuilder opBuilder = scope.graph().opBuilder("MatrixSolveLs", scope.makeOpName("MatrixSolveLs"));
    opBuilder.addInput(matrix.asOutput());
    opBuilder.addInput(rhs.asOutput());
    opBuilder.addInput(l2Regularizer.asOutput());
    return new MatrixSolveLs<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new MatrixSolveLs operation to the graph.
   * 
   * @param scope Current graph scope
   * @param matrix
   * @param rhs
   * @param l2Regularizer
   * @param options an object holding optional attributes values
   * @return a new instance of MatrixSolveLs
   **/
  public static <T> MatrixSolveLs<T> create(Scope scope, Operand<T> matrix, Operand<T> rhs, Operand<Double> l2Regularizer, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("MatrixSolveLs", scope.makeOpName("MatrixSolveLs"));
    opBuilder.addInput(matrix.asOutput());
    opBuilder.addInput(rhs.asOutput());
    opBuilder.addInput(l2Regularizer.asOutput());
    if (options.fast != null) {
      opBuilder.setAttr("fast", options.fast);
    }
    return new MatrixSolveLs<T>(opBuilder.build());
  }
  
  /**
   * @param fast
   **/
  public static Options fast(Boolean fast) {
    return new Options().fast(fast);
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
  private MatrixSolveLs(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
