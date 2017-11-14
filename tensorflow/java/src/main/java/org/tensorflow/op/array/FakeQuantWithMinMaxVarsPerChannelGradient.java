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

public final class FakeQuantWithMinMaxVarsPerChannelGradient extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
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
    
    private Integer numBits;
    private Boolean narrowRange;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new FakeQuantWithMinMaxVarsPerChannelGradient operation to the graph.
   * 
   * @param scope Current graph scope
   * @param gradients
   * @param inputs
   * @param min
   * @param max
   * @return a new instance of FakeQuantWithMinMaxVarsPerChannelGradient
   **/
  public static FakeQuantWithMinMaxVarsPerChannelGradient create(Scope scope, Operand<Float> gradients, Operand<Float> inputs, Operand<Float> min, Operand<Float> max) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FakeQuantWithMinMaxVarsPerChannelGradient", scope.makeOpName("FakeQuantWithMinMaxVarsPerChannelGradient"));
    opBuilder.addInput(gradients.asOutput());
    opBuilder.addInput(inputs.asOutput());
    opBuilder.addInput(min.asOutput());
    opBuilder.addInput(max.asOutput());
    return new FakeQuantWithMinMaxVarsPerChannelGradient(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new FakeQuantWithMinMaxVarsPerChannelGradient operation to the graph.
   * 
   * @param scope Current graph scope
   * @param gradients
   * @param inputs
   * @param min
   * @param max
   * @param options an object holding optional attributes values
   * @return a new instance of FakeQuantWithMinMaxVarsPerChannelGradient
   **/
  public static FakeQuantWithMinMaxVarsPerChannelGradient create(Scope scope, Operand<Float> gradients, Operand<Float> inputs, Operand<Float> min, Operand<Float> max, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FakeQuantWithMinMaxVarsPerChannelGradient", scope.makeOpName("FakeQuantWithMinMaxVarsPerChannelGradient"));
    opBuilder.addInput(gradients.asOutput());
    opBuilder.addInput(inputs.asOutput());
    opBuilder.addInput(min.asOutput());
    opBuilder.addInput(max.asOutput());
    if (options.numBits != null) {
      opBuilder.setAttr("numBits", options.numBits);
    }
    if (options.narrowRange != null) {
      opBuilder.setAttr("narrowRange", options.narrowRange);
    }
    return new FakeQuantWithMinMaxVarsPerChannelGradient(opBuilder.build());
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
  
  public Output<Float> backpropsWrtInput() {
    return backpropsWrtInput;
  }
  
  public Output<Float> backpropWrtMin() {
    return backpropWrtMin;
  }
  
  public Output<Float> backpropWrtMax() {
    return backpropWrtMax;
  }
  
  private Output<Float> backpropsWrtInput;
  private Output<Float> backpropWrtMin;
  private Output<Float> backpropWrtMax;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private FakeQuantWithMinMaxVarsPerChannelGradient(Operation operation) {
    super(operation);
    int outputIdx = 0;
    backpropsWrtInput = operation.output(outputIdx++);
    backpropWrtMin = operation.output(outputIdx++);
    backpropWrtMax = operation.output(outputIdx++);
  }
}
