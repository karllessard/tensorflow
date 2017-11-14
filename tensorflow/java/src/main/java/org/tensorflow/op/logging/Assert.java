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
package org.tensorflow.op.logging;

import org.tensorflow.Operand;
import org.tensorflow.op.Operands;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class Assert extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param summarize
     **/
    public Options summarize(Integer summarize) {
      this.summarize = summarize;
      return this;
    }
    
    private Integer summarize;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new Assert operation to the graph.
   * 
   * @param scope Current graph scope
   * @param condition
   * @param data
   * @return a new instance of Assert
   **/
  public static Assert create(Scope scope, Operand<Boolean> condition, Iterable<Operand<?>> data) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Assert", scope.makeOpName("Assert"));
    opBuilder.addInput(condition.asOutput());
    opBuilder.addInputList(Operands.asOutputs(data));
    return new Assert(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new Assert operation to the graph.
   * 
   * @param scope Current graph scope
   * @param condition
   * @param data
   * @param options an object holding optional attributes values
   * @return a new instance of Assert
   **/
  public static Assert create(Scope scope, Operand<Boolean> condition, Iterable<Operand<?>> data, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Assert", scope.makeOpName("Assert"));
    opBuilder.addInput(condition.asOutput());
    opBuilder.addInputList(Operands.asOutputs(data));
    if (options.summarize != null) {
      opBuilder.setAttr("summarize", options.summarize);
    }
    return new Assert(opBuilder.build());
  }
  
  /**
   * @param summarize
   **/
  public static Options summarize(Integer summarize) {
    return new Options().summarize(summarize);
  }
  
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private Assert(Operation operation) {
    super(operation);
    int outputIdx = 0;
  }
}
