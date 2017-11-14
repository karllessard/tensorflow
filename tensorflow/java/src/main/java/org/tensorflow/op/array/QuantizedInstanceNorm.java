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

public final class QuantizedInstanceNorm<T> extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param outputRangeGiven
     **/
    public Options outputRangeGiven(Boolean outputRangeGiven) {
      this.outputRangeGiven = outputRangeGiven;
      return this;
    }
    
    /**
     * @param givenYMin
     **/
    public Options givenYMin(Float givenYMin) {
      this.givenYMin = givenYMin;
      return this;
    }
    
    /**
     * @param givenYMax
     **/
    public Options givenYMax(Float givenYMax) {
      this.givenYMax = givenYMax;
      return this;
    }
    
    /**
     * @param varianceEpsilon
     **/
    public Options varianceEpsilon(Float varianceEpsilon) {
      this.varianceEpsilon = varianceEpsilon;
      return this;
    }
    
    /**
     * @param minSeparation
     **/
    public Options minSeparation(Float minSeparation) {
      this.minSeparation = minSeparation;
      return this;
    }
    
    private Boolean outputRangeGiven;
    private Float givenYMin;
    private Float givenYMax;
    private Float varianceEpsilon;
    private Float minSeparation;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new QuantizedInstanceNorm operation to the graph.
   * 
   * @param scope Current graph scope
   * @param x
   * @param xMin
   * @param xMax
   * @return a new instance of QuantizedInstanceNorm
   **/
  public static <T> QuantizedInstanceNorm<T> create(Scope scope, Operand<T> x, Operand<Float> xMin, Operand<Float> xMax) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizedInstanceNorm", scope.makeOpName("QuantizedInstanceNorm"));
    opBuilder.addInput(x.asOutput());
    opBuilder.addInput(xMin.asOutput());
    opBuilder.addInput(xMax.asOutput());
    return new QuantizedInstanceNorm<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new QuantizedInstanceNorm operation to the graph.
   * 
   * @param scope Current graph scope
   * @param x
   * @param xMin
   * @param xMax
   * @param options an object holding optional attributes values
   * @return a new instance of QuantizedInstanceNorm
   **/
  public static <T> QuantizedInstanceNorm<T> create(Scope scope, Operand<T> x, Operand<Float> xMin, Operand<Float> xMax, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizedInstanceNorm", scope.makeOpName("QuantizedInstanceNorm"));
    opBuilder.addInput(x.asOutput());
    opBuilder.addInput(xMin.asOutput());
    opBuilder.addInput(xMax.asOutput());
    if (options.outputRangeGiven != null) {
      opBuilder.setAttr("outputRangeGiven", options.outputRangeGiven);
    }
    if (options.givenYMin != null) {
      opBuilder.setAttr("givenYMin", options.givenYMin);
    }
    if (options.givenYMax != null) {
      opBuilder.setAttr("givenYMax", options.givenYMax);
    }
    if (options.varianceEpsilon != null) {
      opBuilder.setAttr("varianceEpsilon", options.varianceEpsilon);
    }
    if (options.minSeparation != null) {
      opBuilder.setAttr("minSeparation", options.minSeparation);
    }
    return new QuantizedInstanceNorm<T>(opBuilder.build());
  }
  
  /**
   * @param outputRangeGiven
   **/
  public static Options outputRangeGiven(Boolean outputRangeGiven) {
    return new Options().outputRangeGiven(outputRangeGiven);
  }
  
  /**
   * @param givenYMin
   **/
  public static Options givenYMin(Float givenYMin) {
    return new Options().givenYMin(givenYMin);
  }
  
  /**
   * @param givenYMax
   **/
  public static Options givenYMax(Float givenYMax) {
    return new Options().givenYMax(givenYMax);
  }
  
  /**
   * @param varianceEpsilon
   **/
  public static Options varianceEpsilon(Float varianceEpsilon) {
    return new Options().varianceEpsilon(varianceEpsilon);
  }
  
  /**
   * @param minSeparation
   **/
  public static Options minSeparation(Float minSeparation) {
    return new Options().minSeparation(minSeparation);
  }
  
  public Output<T> y() {
    return y;
  }
  
  public Output<Float> yMin() {
    return yMin;
  }
  
  public Output<Float> yMax() {
    return yMax;
  }
  
  private Output<T> y;
  private Output<Float> yMin;
  private Output<Float> yMax;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private QuantizedInstanceNorm(Operation operation) {
    super(operation);
    int outputIdx = 0;
    y = operation.output(outputIdx++);
    yMin = operation.output(outputIdx++);
    yMax = operation.output(outputIdx++);
  }
}
