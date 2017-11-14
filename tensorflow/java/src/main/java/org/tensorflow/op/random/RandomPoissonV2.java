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

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class RandomPoissonV2<V> extends PrimitiveOp implements Operand<V> {
  
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
   * Factory method to create a class to wrap a new RandomPoissonV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param shape
   * @param rate
   * @param dtype
   * @return a new instance of RandomPoissonV2
   **/
  public static <T, U, V> RandomPoissonV2<V> create(Scope scope, Operand<T> shape, Operand<U> rate, Class<V> dtype) {
    OperationBuilder opBuilder = scope.graph().opBuilder("RandomPoissonV2", scope.makeOpName("RandomPoissonV2"));
    opBuilder.addInput(shape.asOutput());
    opBuilder.addInput(rate.asOutput());
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    return new RandomPoissonV2<V>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new RandomPoissonV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param shape
   * @param rate
   * @param dtype
   * @param options an object holding optional attributes values
   * @return a new instance of RandomPoissonV2
   **/
  public static <T, U, V> RandomPoissonV2<V> create(Scope scope, Operand<T> shape, Operand<U> rate, Class<V> dtype, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("RandomPoissonV2", scope.makeOpName("RandomPoissonV2"));
    opBuilder.addInput(shape.asOutput());
    opBuilder.addInput(rate.asOutput());
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    if (options.seed != null) {
      opBuilder.setAttr("seed", options.seed);
    }
    if (options.seed2 != null) {
      opBuilder.setAttr("seed2", options.seed2);
    }
    return new RandomPoissonV2<V>(opBuilder.build());
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
  
  public Output<V> output() {
    return output;
  }
  
  @Override
  public Output<V> asOutput() {
    return output;
  }
  
  private Output<V> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private RandomPoissonV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
