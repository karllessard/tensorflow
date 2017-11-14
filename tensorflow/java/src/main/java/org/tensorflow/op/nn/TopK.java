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

public final class TopK<T> extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param sorted
     **/
    public Options sorted(Boolean sorted) {
      this.sorted = sorted;
      return this;
    }
    
    private Boolean sorted;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new TopK operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param k
   * @return a new instance of TopK
   **/
  public static <T> TopK<T> create(Scope scope, Operand<T> input, Integer k) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TopK", scope.makeOpName("TopK"));
    opBuilder.addInput(input.asOutput());
    opBuilder.setAttr("k", k);
    return new TopK<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new TopK operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param k
   * @param options an object holding optional attributes values
   * @return a new instance of TopK
   **/
  public static <T> TopK<T> create(Scope scope, Operand<T> input, Integer k, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TopK", scope.makeOpName("TopK"));
    opBuilder.addInput(input.asOutput());
    opBuilder.setAttr("k", k);
    if (options.sorted != null) {
      opBuilder.setAttr("sorted", options.sorted);
    }
    return new TopK<T>(opBuilder.build());
  }
  
  /**
   * @param sorted
   **/
  public static Options sorted(Boolean sorted) {
    return new Options().sorted(sorted);
  }
  
  public Output<T> values() {
    return values;
  }
  
  public Output<Integer> indices() {
    return indices;
  }
  
  private Output<T> values;
  private Output<Integer> indices;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private TopK(Operation operation) {
    super(operation);
    int outputIdx = 0;
    values = operation.output(outputIdx++);
    indices = operation.output(outputIdx++);
  }
}
