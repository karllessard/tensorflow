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

public final class AddManySparseToTensorsMap extends PrimitiveOp implements Operand<Long> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param container
     **/
    public Options container(String container) {
      this.container = container;
      return this;
    }
    
    /**
     * @param sharedName
     **/
    public Options sharedName(String sharedName) {
      this.sharedName = sharedName;
      return this;
    }
    
    private String container;
    private String sharedName;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new AddManySparseToTensorsMap operation to the graph.
   * 
   * @param scope Current graph scope
   * @param sparseIndices
   * @param sparseValues
   * @param sparseShape
   * @return a new instance of AddManySparseToTensorsMap
   **/
  public static <T> AddManySparseToTensorsMap create(Scope scope, Operand<Long> sparseIndices, Operand<T> sparseValues, Operand<Long> sparseShape) {
    OperationBuilder opBuilder = scope.graph().opBuilder("AddManySparseToTensorsMap", scope.makeOpName("AddManySparseToTensorsMap"));
    opBuilder.addInput(sparseIndices.asOutput());
    opBuilder.addInput(sparseValues.asOutput());
    opBuilder.addInput(sparseShape.asOutput());
    return new AddManySparseToTensorsMap(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new AddManySparseToTensorsMap operation to the graph.
   * 
   * @param scope Current graph scope
   * @param sparseIndices
   * @param sparseValues
   * @param sparseShape
   * @param options an object holding optional attributes values
   * @return a new instance of AddManySparseToTensorsMap
   **/
  public static <T> AddManySparseToTensorsMap create(Scope scope, Operand<Long> sparseIndices, Operand<T> sparseValues, Operand<Long> sparseShape, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("AddManySparseToTensorsMap", scope.makeOpName("AddManySparseToTensorsMap"));
    opBuilder.addInput(sparseIndices.asOutput());
    opBuilder.addInput(sparseValues.asOutput());
    opBuilder.addInput(sparseShape.asOutput());
    if (options.container != null) {
      opBuilder.setAttr("container", options.container);
    }
    if (options.sharedName != null) {
      opBuilder.setAttr("sharedName", options.sharedName);
    }
    return new AddManySparseToTensorsMap(opBuilder.build());
  }
  
  /**
   * @param container
   **/
  public static Options container(String container) {
    return new Options().container(container);
  }
  
  /**
   * @param sharedName
   **/
  public static Options sharedName(String sharedName) {
    return new Options().sharedName(sharedName);
  }
  
  public Output<Long> sparseHandles() {
    return sparseHandles;
  }
  
  @Override
  public Output<Long> asOutput() {
    return sparseHandles;
  }
  
  private Output<Long> sparseHandles;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private AddManySparseToTensorsMap(Operation operation) {
    super(operation);
    int outputIdx = 0;
    sparseHandles = operation.output(outputIdx++);
  }
}
