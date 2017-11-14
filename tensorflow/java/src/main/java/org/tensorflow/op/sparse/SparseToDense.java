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

public final class SparseToDense<U> extends PrimitiveOp implements Operand<U> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param validateIndices
     **/
    public Options validateIndices(Boolean validateIndices) {
      this.validateIndices = validateIndices;
      return this;
    }
    
    private Boolean validateIndices;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new SparseToDense operation to the graph.
   * 
   * @param scope Current graph scope
   * @param sparseIndices
   * @param outputShape
   * @param sparseValues
   * @param defaultValue
   * @return a new instance of SparseToDense
   **/
  public static <T, U> SparseToDense<U> create(Scope scope, Operand<T> sparseIndices, Operand<T> outputShape, Operand<U> sparseValues, Operand<U> defaultValue) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseToDense", scope.makeOpName("SparseToDense"));
    opBuilder.addInput(sparseIndices.asOutput());
    opBuilder.addInput(outputShape.asOutput());
    opBuilder.addInput(sparseValues.asOutput());
    opBuilder.addInput(defaultValue.asOutput());
    return new SparseToDense<U>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new SparseToDense operation to the graph.
   * 
   * @param scope Current graph scope
   * @param sparseIndices
   * @param outputShape
   * @param sparseValues
   * @param defaultValue
   * @param options an object holding optional attributes values
   * @return a new instance of SparseToDense
   **/
  public static <T, U> SparseToDense<U> create(Scope scope, Operand<T> sparseIndices, Operand<T> outputShape, Operand<U> sparseValues, Operand<U> defaultValue, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseToDense", scope.makeOpName("SparseToDense"));
    opBuilder.addInput(sparseIndices.asOutput());
    opBuilder.addInput(outputShape.asOutput());
    opBuilder.addInput(sparseValues.asOutput());
    opBuilder.addInput(defaultValue.asOutput());
    if (options.validateIndices != null) {
      opBuilder.setAttr("validateIndices", options.validateIndices);
    }
    return new SparseToDense<U>(opBuilder.build());
  }
  
  /**
   * @param validateIndices
   **/
  public static Options validateIndices(Boolean validateIndices) {
    return new Options().validateIndices(validateIndices);
  }
  
  public Output<U> dense() {
    return dense;
  }
  
  @Override
  public Output<U> asOutput() {
    return dense;
  }
  
  private Output<U> dense;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private SparseToDense(Operation operation) {
    super(operation);
    int outputIdx = 0;
    dense = operation.output(outputIdx++);
  }
}
