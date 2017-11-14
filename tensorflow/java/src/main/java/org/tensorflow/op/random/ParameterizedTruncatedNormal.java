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
package org.tensorflow.op.random;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class ParameterizedTruncatedNormal<U> extends PrimitiveOp implements Operand<U> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param seed
     **/
    public Options seed(Integer seed) {
      this.seed = seed;
      return this;
    }
    
    /**
     * @param seed2
     **/
    public Options seed2(Integer seed2) {
      this.seed2 = seed2;
      return this;
    }
    
    private Integer seed;
    private Integer seed2;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new ParameterizedTruncatedNormal operation to the graph.
   * 
   * @param scope Current graph scope
   * @param shape
   * @param means
   * @param stdevs
   * @param minvals
   * @param maxvals
   * @return a new instance of ParameterizedTruncatedNormal
   **/
  public static <T, U> ParameterizedTruncatedNormal<U> create(Scope scope, Operand<T> shape, Operand<U> means, Operand<U> stdevs, Operand<U> minvals, Operand<U> maxvals) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ParameterizedTruncatedNormal", scope.makeOpName("ParameterizedTruncatedNormal"));
    opBuilder.addInput(shape.asOutput());
    opBuilder.addInput(means.asOutput());
    opBuilder.addInput(stdevs.asOutput());
    opBuilder.addInput(minvals.asOutput());
    opBuilder.addInput(maxvals.asOutput());
    return new ParameterizedTruncatedNormal<U>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new ParameterizedTruncatedNormal operation to the graph.
   * 
   * @param scope Current graph scope
   * @param shape
   * @param means
   * @param stdevs
   * @param minvals
   * @param maxvals
   * @param options an object holding optional attributes values
   * @return a new instance of ParameterizedTruncatedNormal
   **/
  public static <T, U> ParameterizedTruncatedNormal<U> create(Scope scope, Operand<T> shape, Operand<U> means, Operand<U> stdevs, Operand<U> minvals, Operand<U> maxvals, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ParameterizedTruncatedNormal", scope.makeOpName("ParameterizedTruncatedNormal"));
    opBuilder.addInput(shape.asOutput());
    opBuilder.addInput(means.asOutput());
    opBuilder.addInput(stdevs.asOutput());
    opBuilder.addInput(minvals.asOutput());
    opBuilder.addInput(maxvals.asOutput());
    if (options.seed != null) {
      opBuilder.setAttr("seed", options.seed);
    }
    if (options.seed2 != null) {
      opBuilder.setAttr("seed2", options.seed2);
    }
    return new ParameterizedTruncatedNormal<U>(opBuilder.build());
  }
  
  /**
   * @param seed
   **/
  public static Options seed(Integer seed) {
    return new Options().seed(seed);
  }
  
  /**
   * @param seed2
   **/
  public static Options seed2(Integer seed2) {
    return new Options().seed2(seed2);
  }
  
  public Output<U> output() {
    return output;
  }
  
  @Override
  public Output<U> asOutput() {
    return output;
  }
  
  private Output<U> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ParameterizedTruncatedNormal(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
