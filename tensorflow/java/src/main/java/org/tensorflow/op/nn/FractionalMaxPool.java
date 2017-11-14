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

import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class FractionalMaxPool<T> extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param pseudoRandom
     **/
    public Options pseudoRandom(Boolean pseudoRandom) {
      this.pseudoRandom = pseudoRandom;
      return this;
    }
    
    /**
     * @param overlapping
     **/
    public Options overlapping(Boolean overlapping) {
      this.overlapping = overlapping;
      return this;
    }
    
    /**
     * @param deterministic
     **/
    public Options deterministic(Boolean deterministic) {
      this.deterministic = deterministic;
      return this;
    }
    
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
    
    private Boolean pseudoRandom;
    private Boolean overlapping;
    private Boolean deterministic;
    private Integer seed;
    private Integer seed2;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new FractionalMaxPool operation to the graph.
   * 
   * @param scope Current graph scope
   * @param value
   * @param poolingRatio
   * @return a new instance of FractionalMaxPool
   **/
  public static <T> FractionalMaxPool<T> create(Scope scope, Operand<T> value, List<Float> poolingRatio) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FractionalMaxPool", scope.makeOpName("FractionalMaxPool"));
    opBuilder.addInput(value.asOutput());
    float[] poolingRatioArray = new float[poolingRatio.size()];
    for (int i = 0; i < poolingRatioArray.length; ++i) {
      poolingRatioArray[i] = poolingRatio.get(i);
    }
    opBuilder.setAttr("poolingRatio", poolingRatioArray);
    return new FractionalMaxPool<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new FractionalMaxPool operation to the graph.
   * 
   * @param scope Current graph scope
   * @param value
   * @param poolingRatio
   * @param options an object holding optional attributes values
   * @return a new instance of FractionalMaxPool
   **/
  public static <T> FractionalMaxPool<T> create(Scope scope, Operand<T> value, List<Float> poolingRatio, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FractionalMaxPool", scope.makeOpName("FractionalMaxPool"));
    opBuilder.addInput(value.asOutput());
    float[] poolingRatioArray = new float[poolingRatio.size()];
    for (int i = 0; i < poolingRatioArray.length; ++i) {
      poolingRatioArray[i] = poolingRatio.get(i);
    }
    opBuilder.setAttr("poolingRatio", poolingRatioArray);
    if (options.pseudoRandom != null) {
      opBuilder.setAttr("pseudoRandom", options.pseudoRandom);
    }
    if (options.overlapping != null) {
      opBuilder.setAttr("overlapping", options.overlapping);
    }
    if (options.deterministic != null) {
      opBuilder.setAttr("deterministic", options.deterministic);
    }
    if (options.seed != null) {
      opBuilder.setAttr("seed", options.seed);
    }
    if (options.seed2 != null) {
      opBuilder.setAttr("seed2", options.seed2);
    }
    return new FractionalMaxPool<T>(opBuilder.build());
  }
  
  /**
   * @param pseudoRandom
   **/
  public static Options pseudoRandom(Boolean pseudoRandom) {
    return new Options().pseudoRandom(pseudoRandom);
  }
  
  /**
   * @param overlapping
   **/
  public static Options overlapping(Boolean overlapping) {
    return new Options().overlapping(overlapping);
  }
  
  /**
   * @param deterministic
   **/
  public static Options deterministic(Boolean deterministic) {
    return new Options().deterministic(deterministic);
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
  
  public Output<T> output() {
    return output;
  }
  
  public Output<Long> rowPoolingSequence() {
    return rowPoolingSequence;
  }
  
  public Output<Long> colPoolingSequence() {
    return colPoolingSequence;
  }
  
  private Output<T> output;
  private Output<Long> rowPoolingSequence;
  private Output<Long> colPoolingSequence;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private FractionalMaxPool(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
    rowPoolingSequence = operation.output(outputIdx++);
    colPoolingSequence = operation.output(outputIdx++);
  }
}
