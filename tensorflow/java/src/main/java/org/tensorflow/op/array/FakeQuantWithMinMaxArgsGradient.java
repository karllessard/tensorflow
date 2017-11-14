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
package org.tensorflow.op.array;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class FakeQuantWithMinMaxArgsGradient extends PrimitiveOp implements Operand<Float> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param min
     **/
    public Options min(Float min) {
      this.min = min;
      return this;
    }
    
    /**
     * @param max
     **/
    public Options max(Float max) {
      this.max = max;
      return this;
    }
    
    /**
     * @param numBits
     **/
    public Options numBits(Integer numBits) {
      this.numBits = numBits;
      return this;
    }
    
    /**
     * @param narrowRange
     **/
    public Options narrowRange(Boolean narrowRange) {
      this.narrowRange = narrowRange;
      return this;
    }
    
    private Float min;
    private Float max;
    private Integer numBits;
    private Boolean narrowRange;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new FakeQuantWithMinMaxArgsGradient operation to the graph.
   * 
   * @param scope Current graph scope
   * @param gradients
   * @param inputs
   * @return a new instance of FakeQuantWithMinMaxArgsGradient
   **/
  public static FakeQuantWithMinMaxArgsGradient create(Scope scope, Operand<Float> gradients, Operand<Float> inputs) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FakeQuantWithMinMaxArgsGradient", scope.makeOpName("FakeQuantWithMinMaxArgsGradient"));
    opBuilder.addInput(gradients.asOutput());
    opBuilder.addInput(inputs.asOutput());
    return new FakeQuantWithMinMaxArgsGradient(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new FakeQuantWithMinMaxArgsGradient operation to the graph.
   * 
   * @param scope Current graph scope
   * @param gradients
   * @param inputs
   * @param options an object holding optional attributes values
   * @return a new instance of FakeQuantWithMinMaxArgsGradient
   **/
  public static FakeQuantWithMinMaxArgsGradient create(Scope scope, Operand<Float> gradients, Operand<Float> inputs, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FakeQuantWithMinMaxArgsGradient", scope.makeOpName("FakeQuantWithMinMaxArgsGradient"));
    opBuilder.addInput(gradients.asOutput());
    opBuilder.addInput(inputs.asOutput());
    if (options.min != null) {
      opBuilder.setAttr("min", options.min);
    }
    if (options.max != null) {
      opBuilder.setAttr("max", options.max);
    }
    if (options.numBits != null) {
      opBuilder.setAttr("numBits", options.numBits);
    }
    if (options.narrowRange != null) {
      opBuilder.setAttr("narrowRange", options.narrowRange);
    }
    return new FakeQuantWithMinMaxArgsGradient(opBuilder.build());
  }
  
  /**
   * @param min
   **/
  public static Options min(Float min) {
    return new Options().min(min);
  }
  
  /**
   * @param max
   **/
  public static Options max(Float max) {
    return new Options().max(max);
  }
  
  /**
   * @param numBits
   **/
  public static Options numBits(Integer numBits) {
    return new Options().numBits(numBits);
  }
  
  /**
   * @param narrowRange
   **/
  public static Options narrowRange(Boolean narrowRange) {
    return new Options().narrowRange(narrowRange);
  }
  
  public Output<Float> backprops() {
    return backprops;
  }
  
  @Override
  public Output<Float> asOutput() {
    return backprops;
  }
  
  private Output<Float> backprops;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private FakeQuantWithMinMaxArgsGradient(Operation operation) {
    super(operation);
    int outputIdx = 0;
    backprops = operation.output(outputIdx++);
  }
}
