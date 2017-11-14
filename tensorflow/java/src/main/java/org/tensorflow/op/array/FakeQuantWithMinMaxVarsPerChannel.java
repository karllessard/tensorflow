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

public final class FakeQuantWithMinMaxVarsPerChannel extends PrimitiveOp implements Operand<Float> {
  
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
   * Factory method to create a class to wrap a new FakeQuantWithMinMaxVarsPerChannel operation to the graph.
   * 
   * @param scope Current graph scope
   * @param inputs
   * @param min
   * @param max
   * @return a new instance of FakeQuantWithMinMaxVarsPerChannel
   **/
  public static FakeQuantWithMinMaxVarsPerChannel create(Scope scope, Operand<Float> inputs, Operand<Float> min, Operand<Float> max) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FakeQuantWithMinMaxVarsPerChannel", scope.makeOpName("FakeQuantWithMinMaxVarsPerChannel"));
    opBuilder.addInput(inputs.asOutput());
    opBuilder.addInput(min.asOutput());
    opBuilder.addInput(max.asOutput());
    return new FakeQuantWithMinMaxVarsPerChannel(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new FakeQuantWithMinMaxVarsPerChannel operation to the graph.
   * 
   * @param scope Current graph scope
   * @param inputs
   * @param min
   * @param max
   * @param options an object holding optional attributes values
   * @return a new instance of FakeQuantWithMinMaxVarsPerChannel
   **/
  public static FakeQuantWithMinMaxVarsPerChannel create(Scope scope, Operand<Float> inputs, Operand<Float> min, Operand<Float> max, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FakeQuantWithMinMaxVarsPerChannel", scope.makeOpName("FakeQuantWithMinMaxVarsPerChannel"));
    opBuilder.addInput(inputs.asOutput());
    opBuilder.addInput(min.asOutput());
    opBuilder.addInput(max.asOutput());
    if (options.numBits != null) {
      opBuilder.setAttr("numBits", options.numBits);
    }
    if (options.narrowRange != null) {
      opBuilder.setAttr("narrowRange", options.narrowRange);
    }
    return new FakeQuantWithMinMaxVarsPerChannel(opBuilder.build());
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
  
  public Output<Float> outputs() {
    return outputs;
  }
  
  @Override
  public Output<Float> asOutput() {
    return outputs;
  }
  
  private Output<Float> outputs;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private FakeQuantWithMinMaxVarsPerChannel(Operation operation) {
    super(operation);
    int outputIdx = 0;
    outputs = operation.output(outputIdx++);
  }
}
