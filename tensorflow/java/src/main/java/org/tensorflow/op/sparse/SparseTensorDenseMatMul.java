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

public final class SparseTensorDenseMatMul<U> extends PrimitiveOp implements Operand<U> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param adjointA
     **/
    public Options adjointA(Boolean adjointA) {
      this.adjointA = adjointA;
      return this;
    }
    
    /**
     * @param adjointB
     **/
    public Options adjointB(Boolean adjointB) {
      this.adjointB = adjointB;
      return this;
    }
    
    private Boolean adjointA;
    private Boolean adjointB;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new SparseTensorDenseMatMul operation to the graph.
   * 
   * @param scope Current graph scope
   * @param aIndices
   * @param aValues
   * @param aShape
   * @param b
   * @return a new instance of SparseTensorDenseMatMul
   **/
  public static <T, U> SparseTensorDenseMatMul<U> create(Scope scope, Operand<T> aIndices, Operand<U> aValues, Operand<Long> aShape, Operand<U> b) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseTensorDenseMatMul", scope.makeOpName("SparseTensorDenseMatMul"));
    opBuilder.addInput(aIndices.asOutput());
    opBuilder.addInput(aValues.asOutput());
    opBuilder.addInput(aShape.asOutput());
    opBuilder.addInput(b.asOutput());
    return new SparseTensorDenseMatMul<U>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new SparseTensorDenseMatMul operation to the graph.
   * 
   * @param scope Current graph scope
   * @param aIndices
   * @param aValues
   * @param aShape
   * @param b
   * @param options an object holding optional attributes values
   * @return a new instance of SparseTensorDenseMatMul
   **/
  public static <T, U> SparseTensorDenseMatMul<U> create(Scope scope, Operand<T> aIndices, Operand<U> aValues, Operand<Long> aShape, Operand<U> b, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseTensorDenseMatMul", scope.makeOpName("SparseTensorDenseMatMul"));
    opBuilder.addInput(aIndices.asOutput());
    opBuilder.addInput(aValues.asOutput());
    opBuilder.addInput(aShape.asOutput());
    opBuilder.addInput(b.asOutput());
    if (options.adjointA != null) {
      opBuilder.setAttr("adjointA", options.adjointA);
    }
    if (options.adjointB != null) {
      opBuilder.setAttr("adjointB", options.adjointB);
    }
    return new SparseTensorDenseMatMul<U>(opBuilder.build());
  }
  
  /**
   * @param adjointA
   **/
  public static Options adjointA(Boolean adjointA) {
    return new Options().adjointA(adjointA);
  }
  
  /**
   * @param adjointB
   **/
  public static Options adjointB(Boolean adjointB) {
    return new Options().adjointB(adjointB);
  }
  
  public Output<U> product() {
    return product;
  }
  
  @Override
  public Output<U> asOutput() {
    return product;
  }
  
  private Output<U> product;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private SparseTensorDenseMatMul(Operation operation) {
    super(operation);
    int outputIdx = 0;
    product = operation.output(outputIdx++);
  }
}
