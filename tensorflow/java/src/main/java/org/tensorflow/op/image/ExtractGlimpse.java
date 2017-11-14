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
package org.tensorflow.op.image;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class ExtractGlimpse extends PrimitiveOp implements Operand<Float> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param centered
     **/
    public Options centered(Boolean centered) {
      this.centered = centered;
      return this;
    }
    
    /**
     * @param normalized
     **/
    public Options normalized(Boolean normalized) {
      this.normalized = normalized;
      return this;
    }
    
    /**
     * @param uniformNoise
     **/
    public Options uniformNoise(Boolean uniformNoise) {
      this.uniformNoise = uniformNoise;
      return this;
    }
    
    private Boolean centered;
    private Boolean normalized;
    private Boolean uniformNoise;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new ExtractGlimpse operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param size
   * @param offsets
   * @return a new instance of ExtractGlimpse
   **/
  public static ExtractGlimpse create(Scope scope, Operand<Float> input, Operand<Integer> size, Operand<Float> offsets) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ExtractGlimpse", scope.makeOpName("ExtractGlimpse"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(size.asOutput());
    opBuilder.addInput(offsets.asOutput());
    return new ExtractGlimpse(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new ExtractGlimpse operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param size
   * @param offsets
   * @param options an object holding optional attributes values
   * @return a new instance of ExtractGlimpse
   **/
  public static ExtractGlimpse create(Scope scope, Operand<Float> input, Operand<Integer> size, Operand<Float> offsets, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ExtractGlimpse", scope.makeOpName("ExtractGlimpse"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(size.asOutput());
    opBuilder.addInput(offsets.asOutput());
    if (options.centered != null) {
      opBuilder.setAttr("centered", options.centered);
    }
    if (options.normalized != null) {
      opBuilder.setAttr("normalized", options.normalized);
    }
    if (options.uniformNoise != null) {
      opBuilder.setAttr("uniformNoise", options.uniformNoise);
    }
    return new ExtractGlimpse(opBuilder.build());
  }
  
  /**
   * @param centered
   **/
  public static Options centered(Boolean centered) {
    return new Options().centered(centered);
  }
  
  /**
   * @param normalized
   **/
  public static Options normalized(Boolean normalized) {
    return new Options().normalized(normalized);
  }
  
  /**
   * @param uniformNoise
   **/
  public static Options uniformNoise(Boolean uniformNoise) {
    return new Options().uniformNoise(uniformNoise);
  }
  
  public Output<Float> glimpse() {
    return glimpse;
  }
  
  @Override
  public Output<Float> asOutput() {
    return glimpse;
  }
  
  private Output<Float> glimpse;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ExtractGlimpse(Operation operation) {
    super(operation);
    int outputIdx = 0;
    glimpse = operation.output(outputIdx++);
  }
}
