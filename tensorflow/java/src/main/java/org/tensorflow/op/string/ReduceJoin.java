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
package org.tensorflow.op.string;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class ReduceJoin extends PrimitiveOp implements Operand<String> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param keepDims
     **/
    public Options keepDims(Boolean keepDims) {
      this.keepDims = keepDims;
      return this;
    }
    
    /**
     * @param separator
     **/
    public Options separator(String separator) {
      this.separator = separator;
      return this;
    }
    
    private Boolean keepDims;
    private String separator;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new ReduceJoin operation to the graph.
   * 
   * @param scope Current graph scope
   * @param inputs
   * @param reductionIndices
   * @return a new instance of ReduceJoin
   **/
  public static ReduceJoin create(Scope scope, Operand<String> inputs, Operand<Integer> reductionIndices) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ReduceJoin", scope.makeOpName("ReduceJoin"));
    opBuilder.addInput(inputs.asOutput());
    opBuilder.addInput(reductionIndices.asOutput());
    return new ReduceJoin(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new ReduceJoin operation to the graph.
   * 
   * @param scope Current graph scope
   * @param inputs
   * @param reductionIndices
   * @param options an object holding optional attributes values
   * @return a new instance of ReduceJoin
   **/
  public static ReduceJoin create(Scope scope, Operand<String> inputs, Operand<Integer> reductionIndices, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ReduceJoin", scope.makeOpName("ReduceJoin"));
    opBuilder.addInput(inputs.asOutput());
    opBuilder.addInput(reductionIndices.asOutput());
    if (options.keepDims != null) {
      opBuilder.setAttr("keepDims", options.keepDims);
    }
    if (options.separator != null) {
      opBuilder.setAttr("separator", options.separator);
    }
    return new ReduceJoin(opBuilder.build());
  }
  
  /**
   * @param keepDims
   **/
  public static Options keepDims(Boolean keepDims) {
    return new Options().keepDims(keepDims);
  }
  
  /**
   * @param separator
   **/
  public static Options separator(String separator) {
    return new Options().separator(separator);
  }
  
  public Output<String> output() {
    return output;
  }
  
  @Override
  public Output<String> asOutput() {
    return output;
  }
  
  private Output<String> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ReduceJoin(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
