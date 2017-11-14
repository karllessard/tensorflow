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
package org.tensorflow.op.nn;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class NthElement<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param reverse
     **/
    public Options reverse(Boolean reverse) {
      this.reverse = reverse;
      return this;
    }
    
    private Boolean reverse;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new NthElement operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param n
   * @return a new instance of NthElement
   **/
  public static <T> NthElement<T> create(Scope scope, Operand<T> input, Operand<Integer> n) {
    OperationBuilder opBuilder = scope.graph().opBuilder("NthElement", scope.makeOpName("NthElement"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(n.asOutput());
    return new NthElement<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new NthElement operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param n
   * @param options an object holding optional attributes values
   * @return a new instance of NthElement
   **/
  public static <T> NthElement<T> create(Scope scope, Operand<T> input, Operand<Integer> n, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("NthElement", scope.makeOpName("NthElement"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(n.asOutput());
    if (options.reverse != null) {
      opBuilder.setAttr("reverse", options.reverse);
    }
    return new NthElement<T>(opBuilder.build());
  }
  
  /**
   * @param reverse
   **/
  public static Options reverse(Boolean reverse) {
    return new Options().reverse(reverse);
  }
  
  public Output<T> values() {
    return values;
  }
  
  @Override
  public Output<T> asOutput() {
    return values;
  }
  
  private Output<T> values;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private NthElement(Operation operation) {
    super(operation);
    int outputIdx = 0;
    values = operation.output(outputIdx++);
  }
}
