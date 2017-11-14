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
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class AudioSummaryV2 extends PrimitiveOp implements Operand<String> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param maxOutputs
     **/
    public Options maxOutputs(Integer maxOutputs) {
      this.maxOutputs = maxOutputs;
      return this;
    }
    
    private Integer maxOutputs;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new AudioSummaryV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param tag
   * @param tensor
   * @param sampleRate
   * @return a new instance of AudioSummaryV2
   **/
  public static AudioSummaryV2 create(Scope scope, Operand<String> tag, Operand<Float> tensor, Operand<Float> sampleRate) {
    OperationBuilder opBuilder = scope.graph().opBuilder("AudioSummaryV2", scope.makeOpName("AudioSummaryV2"));
    opBuilder.addInput(tag.asOutput());
    opBuilder.addInput(tensor.asOutput());
    opBuilder.addInput(sampleRate.asOutput());
    return new AudioSummaryV2(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new AudioSummaryV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param tag
   * @param tensor
   * @param sampleRate
   * @param options an object holding optional attributes values
   * @return a new instance of AudioSummaryV2
   **/
  public static AudioSummaryV2 create(Scope scope, Operand<String> tag, Operand<Float> tensor, Operand<Float> sampleRate, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("AudioSummaryV2", scope.makeOpName("AudioSummaryV2"));
    opBuilder.addInput(tag.asOutput());
    opBuilder.addInput(tensor.asOutput());
    opBuilder.addInput(sampleRate.asOutput());
    if (options.maxOutputs != null) {
      opBuilder.setAttr("maxOutputs", options.maxOutputs);
    }
    return new AudioSummaryV2(opBuilder.build());
  }
  
  /**
   * @param maxOutputs
   **/
  public static Options maxOutputs(Integer maxOutputs) {
    return new Options().maxOutputs(maxOutputs);
  }
  
  public Output<String> summary() {
    return summary;
  }
  
  @Override
  public Output<String> asOutput() {
    return summary;
  }
  
  private Output<String> summary;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private AudioSummaryV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    summary = operation.output(outputIdx++);
  }
}
