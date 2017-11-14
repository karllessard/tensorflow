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

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class LRNGrad<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param depthRadius
     **/
    public Options depthRadius(Integer depthRadius) {
      this.depthRadius = depthRadius;
      return this;
    }
    
    /**
     * @param bias
     **/
    public Options bias(Float bias) {
      this.bias = bias;
      return this;
    }
    
    /**
     * @param alpha
     **/
    public Options alpha(Float alpha) {
      this.alpha = alpha;
      return this;
    }
    
    /**
     * @param beta
     **/
    public Options beta(Float beta) {
      this.beta = beta;
      return this;
    }
    
    private Integer depthRadius;
    private Float bias;
    private Float alpha;
    private Float beta;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new LRNGrad operation to the graph.
   * 
   * @param scope Current graph scope
   * @param inputGrads
   * @param inputImage
   * @param outputImage
   * @return a new instance of LRNGrad
   **/
  public static <T> LRNGrad<T> create(Scope scope, Operand<T> inputGrads, Operand<T> inputImage, Operand<T> outputImage) {
    OperationBuilder opBuilder = scope.graph().opBuilder("LRNGrad", scope.makeOpName("LRNGrad"));
    opBuilder.addInput(inputGrads.asOutput());
    opBuilder.addInput(inputImage.asOutput());
    opBuilder.addInput(outputImage.asOutput());
    return new LRNGrad<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new LRNGrad operation to the graph.
   * 
   * @param scope Current graph scope
   * @param inputGrads
   * @param inputImage
   * @param outputImage
   * @param options an object holding optional attributes values
   * @return a new instance of LRNGrad
   **/
  public static <T> LRNGrad<T> create(Scope scope, Operand<T> inputGrads, Operand<T> inputImage, Operand<T> outputImage, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("LRNGrad", scope.makeOpName("LRNGrad"));
    opBuilder.addInput(inputGrads.asOutput());
    opBuilder.addInput(inputImage.asOutput());
    opBuilder.addInput(outputImage.asOutput());
    if (options.depthRadius != null) {
      opBuilder.setAttr("depthRadius", options.depthRadius);
    }
    if (options.bias != null) {
      opBuilder.setAttr("bias", options.bias);
    }
    if (options.alpha != null) {
      opBuilder.setAttr("alpha", options.alpha);
    }
    if (options.beta != null) {
      opBuilder.setAttr("beta", options.beta);
    }
    return new LRNGrad<T>(opBuilder.build());
  }
  
  /**
   * @param depthRadius
   **/
  public static Options depthRadius(Integer depthRadius) {
    return new Options().depthRadius(depthRadius);
  }
  
  /**
   * @param bias
   **/
  public static Options bias(Float bias) {
    return new Options().bias(bias);
  }
  
  /**
   * @param alpha
   **/
  public static Options alpha(Float alpha) {
    return new Options().alpha(alpha);
  }
  
  /**
   * @param beta
   **/
  public static Options beta(Float beta) {
    return new Options().beta(beta);
  }
  
  public Output<T> output() {
    return output;
  }
  
  @Override
  public Output<T> asOutput() {
    return output;
  }
  
  private Output<T> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private LRNGrad(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
