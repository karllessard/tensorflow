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

public final class MatMul<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param transposeA
     **/
    public Options transposeA(Boolean transposeA) {
      this.transposeA = transposeA;
      return this;
    }
    
    /**
     * @param transposeB
     **/
    public Options transposeB(Boolean transposeB) {
      this.transposeB = transposeB;
      return this;
    }
    
    private Boolean transposeA;
    private Boolean transposeB;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new MatMul operation to the graph.
   * 
   * @param scope Current graph scope
   * @param a
   * @param b
   * @return a new instance of MatMul
   **/
  public static <T> MatMul<T> create(Scope scope, Operand<T> a, Operand<T> b) {
    OperationBuilder opBuilder = scope.graph().opBuilder("MatMul", scope.makeOpName("MatMul"));
    opBuilder.addInput(a.asOutput());
    opBuilder.addInput(b.asOutput());
    return new MatMul<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new MatMul operation to the graph.
   * 
   * @param scope Current graph scope
   * @param a
   * @param b
   * @param options an object holding optional attributes values
   * @return a new instance of MatMul
   **/
  public static <T> MatMul<T> create(Scope scope, Operand<T> a, Operand<T> b, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("MatMul", scope.makeOpName("MatMul"));
    opBuilder.addInput(a.asOutput());
    opBuilder.addInput(b.asOutput());
    if (options.transposeA != null) {
      opBuilder.setAttr("transposeA", options.transposeA);
    }
    if (options.transposeB != null) {
      opBuilder.setAttr("transposeB", options.transposeB);
    }
    return new MatMul<T>(opBuilder.build());
  }
  
  /**
   * @param transposeA
   **/
  public static Options transposeA(Boolean transposeA) {
    return new Options().transposeA(transposeA);
  }
  
  /**
   * @param transposeB
   **/
  public static Options transposeB(Boolean transposeB) {
    return new Options().transposeB(transposeB);
  }
  
  public Output<T> product() {
    return product;
  }
  
  @Override
  public Output<T> asOutput() {
    return product;
  }
  
  private Output<T> product;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private MatMul(Operation operation) {
    super(operation);
    int outputIdx = 0;
    product = operation.output(outputIdx++);
  }
}
