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

public final class ApproximateEqual extends PrimitiveOp implements Operand<Boolean> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param tolerance
     **/
    public Options tolerance(Float tolerance) {
      this.tolerance = tolerance;
      return this;
    }
    
    private Float tolerance;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new ApproximateEqual operation to the graph.
   * 
   * @param scope Current graph scope
   * @param x
   * @param y
   * @return a new instance of ApproximateEqual
   **/
  public static <T> ApproximateEqual create(Scope scope, Operand<T> x, Operand<T> y) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ApproximateEqual", scope.makeOpName("ApproximateEqual"));
    opBuilder.addInput(x.asOutput());
    opBuilder.addInput(y.asOutput());
    return new ApproximateEqual(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new ApproximateEqual operation to the graph.
   * 
   * @param scope Current graph scope
   * @param x
   * @param y
   * @param options an object holding optional attributes values
   * @return a new instance of ApproximateEqual
   **/
  public static <T> ApproximateEqual create(Scope scope, Operand<T> x, Operand<T> y, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ApproximateEqual", scope.makeOpName("ApproximateEqual"));
    opBuilder.addInput(x.asOutput());
    opBuilder.addInput(y.asOutput());
    if (options.tolerance != null) {
      opBuilder.setAttr("tolerance", options.tolerance);
    }
    return new ApproximateEqual(opBuilder.build());
  }
  
  /**
   * @param tolerance
   **/
  public static Options tolerance(Float tolerance) {
    return new Options().tolerance(tolerance);
  }
  
  public Output<Boolean> z() {
    return z;
  }
  
  @Override
  public Output<Boolean> asOutput() {
    return z;
  }
  
  private Output<Boolean> z;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ApproximateEqual(Operation operation) {
    super(operation);
    int outputIdx = 0;
    z = operation.output(outputIdx++);
  }
}
