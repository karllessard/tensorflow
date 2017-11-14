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

public final class BatchMatMul<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param adjX
     **/
    public Options adjX(Boolean adjX) {
      this.adjX = adjX;
      return this;
    }
    
    /**
     * @param adjY
     **/
    public Options adjY(Boolean adjY) {
      this.adjY = adjY;
      return this;
    }
    
    private Boolean adjX;
    private Boolean adjY;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new BatchMatMul operation to the graph.
   * 
   * @param scope Current graph scope
   * @param x
   * @param y
   * @return a new instance of BatchMatMul
   **/
  public static <T> BatchMatMul<T> create(Scope scope, Operand<T> x, Operand<T> y) {
    OperationBuilder opBuilder = scope.graph().opBuilder("BatchMatMul", scope.makeOpName("BatchMatMul"));
    opBuilder.addInput(x.asOutput());
    opBuilder.addInput(y.asOutput());
    return new BatchMatMul<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new BatchMatMul operation to the graph.
   * 
   * @param scope Current graph scope
   * @param x
   * @param y
   * @param options an object holding optional attributes values
   * @return a new instance of BatchMatMul
   **/
  public static <T> BatchMatMul<T> create(Scope scope, Operand<T> x, Operand<T> y, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("BatchMatMul", scope.makeOpName("BatchMatMul"));
    opBuilder.addInput(x.asOutput());
    opBuilder.addInput(y.asOutput());
    if (options.adjX != null) {
      opBuilder.setAttr("adjX", options.adjX);
    }
    if (options.adjY != null) {
      opBuilder.setAttr("adjY", options.adjY);
    }
    return new BatchMatMul<T>(opBuilder.build());
  }
  
  /**
   * @param adjX
   **/
  public static Options adjX(Boolean adjX) {
    return new Options().adjX(adjX);
  }
  
  /**
   * @param adjY
   **/
  public static Options adjY(Boolean adjY) {
    return new Options().adjY(adjY);
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
  private BatchMatMul(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
