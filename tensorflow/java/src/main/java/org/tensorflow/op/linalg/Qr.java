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

public final class Qr<T> extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param fullMatrices
     **/
    public Options fullMatrices(Boolean fullMatrices) {
      this.fullMatrices = fullMatrices;
      return this;
    }
    
    private Boolean fullMatrices;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new Qr operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @return a new instance of Qr
   **/
  public static <T> Qr<T> create(Scope scope, Operand<T> input) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Qr", scope.makeOpName("Qr"));
    opBuilder.addInput(input.asOutput());
    return new Qr<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new Qr operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param options an object holding optional attributes values
   * @return a new instance of Qr
   **/
  public static <T> Qr<T> create(Scope scope, Operand<T> input, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Qr", scope.makeOpName("Qr"));
    opBuilder.addInput(input.asOutput());
    if (options.fullMatrices != null) {
      opBuilder.setAttr("fullMatrices", options.fullMatrices);
    }
    return new Qr<T>(opBuilder.build());
  }
  
  /**
   * @param fullMatrices
   **/
  public static Options fullMatrices(Boolean fullMatrices) {
    return new Options().fullMatrices(fullMatrices);
  }
  
  public Output<T> q() {
    return q;
  }
  
  public Output<T> r() {
    return r;
  }
  
  private Output<T> q;
  private Output<T> r;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private Qr(Operation operation) {
    super(operation);
    int outputIdx = 0;
    q = operation.output(outputIdx++);
    r = operation.output(outputIdx++);
  }
}
