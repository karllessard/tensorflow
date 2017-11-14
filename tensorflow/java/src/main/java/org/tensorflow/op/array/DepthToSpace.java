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

public final class DepthToSpace<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param dataFormat
     **/
    public Options dataFormat(String dataFormat) {
      this.dataFormat = dataFormat;
      return this;
    }
    
    private String dataFormat;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new DepthToSpace operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param blockSize
   * @return a new instance of DepthToSpace
   **/
  public static <T> DepthToSpace<T> create(Scope scope, Operand<T> input, Integer blockSize) {
    OperationBuilder opBuilder = scope.graph().opBuilder("DepthToSpace", scope.makeOpName("DepthToSpace"));
    opBuilder.addInput(input.asOutput());
    opBuilder.setAttr("blockSize", blockSize);
    return new DepthToSpace<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new DepthToSpace operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param blockSize
   * @param options an object holding optional attributes values
   * @return a new instance of DepthToSpace
   **/
  public static <T> DepthToSpace<T> create(Scope scope, Operand<T> input, Integer blockSize, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("DepthToSpace", scope.makeOpName("DepthToSpace"));
    opBuilder.addInput(input.asOutput());
    opBuilder.setAttr("blockSize", blockSize);
    if (options.dataFormat != null) {
      opBuilder.setAttr("dataFormat", options.dataFormat);
    }
    return new DepthToSpace<T>(opBuilder.build());
  }
  
  /**
   * @param dataFormat
   **/
  public static Options dataFormat(String dataFormat) {
    return new Options().dataFormat(dataFormat);
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
  private DepthToSpace(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
