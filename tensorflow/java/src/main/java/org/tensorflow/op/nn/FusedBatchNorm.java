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

public final class FusedBatchNorm<T> extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param epsilon
     **/
    public Options epsilon(Float epsilon) {
      this.epsilon = epsilon;
      return this;
    }
    
    /**
     * @param dataFormat
     **/
    public Options dataFormat(String dataFormat) {
      this.dataFormat = dataFormat;
      return this;
    }
    
    /**
     * @param isTraining
     **/
    public Options isTraining(Boolean isTraining) {
      this.isTraining = isTraining;
      return this;
    }
    
    private Float epsilon;
    private String dataFormat;
    private Boolean isTraining;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new FusedBatchNorm operation to the graph.
   * 
   * @param scope Current graph scope
   * @param x
   * @param scale
   * @param offset
   * @param mean
   * @param variance
   * @return a new instance of FusedBatchNorm
   **/
  public static <T> FusedBatchNorm<T> create(Scope scope, Operand<T> x, Operand<T> scale, Operand<T> offset, Operand<T> mean, Operand<T> variance) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FusedBatchNorm", scope.makeOpName("FusedBatchNorm"));
    opBuilder.addInput(x.asOutput());
    opBuilder.addInput(scale.asOutput());
    opBuilder.addInput(offset.asOutput());
    opBuilder.addInput(mean.asOutput());
    opBuilder.addInput(variance.asOutput());
    return new FusedBatchNorm<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new FusedBatchNorm operation to the graph.
   * 
   * @param scope Current graph scope
   * @param x
   * @param scale
   * @param offset
   * @param mean
   * @param variance
   * @param options an object holding optional attributes values
   * @return a new instance of FusedBatchNorm
   **/
  public static <T> FusedBatchNorm<T> create(Scope scope, Operand<T> x, Operand<T> scale, Operand<T> offset, Operand<T> mean, Operand<T> variance, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FusedBatchNorm", scope.makeOpName("FusedBatchNorm"));
    opBuilder.addInput(x.asOutput());
    opBuilder.addInput(scale.asOutput());
    opBuilder.addInput(offset.asOutput());
    opBuilder.addInput(mean.asOutput());
    opBuilder.addInput(variance.asOutput());
    if (options.epsilon != null) {
      opBuilder.setAttr("epsilon", options.epsilon);
    }
    if (options.dataFormat != null) {
      opBuilder.setAttr("dataFormat", options.dataFormat);
    }
    if (options.isTraining != null) {
      opBuilder.setAttr("isTraining", options.isTraining);
    }
    return new FusedBatchNorm<T>(opBuilder.build());
  }
  
  /**
   * @param epsilon
   **/
  public static Options epsilon(Float epsilon) {
    return new Options().epsilon(epsilon);
  }
  
  /**
   * @param dataFormat
   **/
  public static Options dataFormat(String dataFormat) {
    return new Options().dataFormat(dataFormat);
  }
  
  /**
   * @param isTraining
   **/
  public static Options isTraining(Boolean isTraining) {
    return new Options().isTraining(isTraining);
  }
  
  public Output<T> y() {
    return y;
  }
  
  public Output<T> batchMean() {
    return batchMean;
  }
  
  public Output<T> batchVariance() {
    return batchVariance;
  }
  
  public Output<T> reserveSpace1() {
    return reserveSpace1;
  }
  
  public Output<T> reserveSpace2() {
    return reserveSpace2;
  }
  
  private Output<T> y;
  private Output<T> batchMean;
  private Output<T> batchVariance;
  private Output<T> reserveSpace1;
  private Output<T> reserveSpace2;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private FusedBatchNorm(Operation operation) {
    super(operation);
    int outputIdx = 0;
    y = operation.output(outputIdx++);
    batchMean = operation.output(outputIdx++);
    batchVariance = operation.output(outputIdx++);
    reserveSpace1 = operation.output(outputIdx++);
    reserveSpace2 = operation.output(outputIdx++);
  }
}
