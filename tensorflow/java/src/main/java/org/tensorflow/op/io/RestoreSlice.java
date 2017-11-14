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
package org.tensorflow.op.io;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class RestoreSlice<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param preferredShard
     **/
    public Options preferredShard(Integer preferredShard) {
      this.preferredShard = preferredShard;
      return this;
    }
    
    private Integer preferredShard;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new RestoreSlice operation to the graph.
   * 
   * @param scope Current graph scope
   * @param filePattern
   * @param tensorName
   * @param shapeAndSlice
   * @param dt
   * @return a new instance of RestoreSlice
   **/
  public static <T> RestoreSlice<T> create(Scope scope, Operand<String> filePattern, Operand<String> tensorName, Operand<String> shapeAndSlice, Class<T> dt) {
    OperationBuilder opBuilder = scope.graph().opBuilder("RestoreSlice", scope.makeOpName("RestoreSlice"));
    opBuilder.addInput(filePattern.asOutput());
    opBuilder.addInput(tensorName.asOutput());
    opBuilder.addInput(shapeAndSlice.asOutput());
    opBuilder.setAttr("dt", DataType.fromClass(dt));
    return new RestoreSlice<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new RestoreSlice operation to the graph.
   * 
   * @param scope Current graph scope
   * @param filePattern
   * @param tensorName
   * @param shapeAndSlice
   * @param dt
   * @param options an object holding optional attributes values
   * @return a new instance of RestoreSlice
   **/
  public static <T> RestoreSlice<T> create(Scope scope, Operand<String> filePattern, Operand<String> tensorName, Operand<String> shapeAndSlice, Class<T> dt, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("RestoreSlice", scope.makeOpName("RestoreSlice"));
    opBuilder.addInput(filePattern.asOutput());
    opBuilder.addInput(tensorName.asOutput());
    opBuilder.addInput(shapeAndSlice.asOutput());
    opBuilder.setAttr("dt", DataType.fromClass(dt));
    if (options.preferredShard != null) {
      opBuilder.setAttr("preferredShard", options.preferredShard);
    }
    return new RestoreSlice<T>(opBuilder.build());
  }
  
  /**
   * @param preferredShard
   **/
  public static Options preferredShard(Integer preferredShard) {
    return new Options().preferredShard(preferredShard);
  }
  
  public Output<T> tensor() {
    return tensor;
  }
  
  @Override
  public Output<T> asOutput() {
    return tensor;
  }
  
  private Output<T> tensor;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private RestoreSlice(Operation operation) {
    super(operation);
    int outputIdx = 0;
    tensor = operation.output(outputIdx++);
  }
}
