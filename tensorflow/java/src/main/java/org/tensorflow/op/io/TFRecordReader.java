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

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class TFRecordReader extends PrimitiveOp implements Operand<String> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param container
     **/
    public Options container(String container) {
      this.container = container;
      return this;
    }
    
    /**
     * @param sharedName
     **/
    public Options sharedName(String sharedName) {
      this.sharedName = sharedName;
      return this;
    }
    
    /**
     * @param compressionType
     **/
    public Options compressionType(String compressionType) {
      this.compressionType = compressionType;
      return this;
    }
    
    private String container;
    private String sharedName;
    private String compressionType;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new TFRecordReader operation to the graph.
   * 
   * @param scope Current graph scope
   * @return a new instance of TFRecordReader
   **/
  public static TFRecordReader create(Scope scope) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TFRecordReader", scope.makeOpName("TFRecordReader"));
    return new TFRecordReader(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new TFRecordReader operation to the graph.
   * 
   * @param scope Current graph scope
   * @param options an object holding optional attributes values
   * @return a new instance of TFRecordReader
   **/
  public static TFRecordReader create(Scope scope, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TFRecordReader", scope.makeOpName("TFRecordReader"));
    if (options.container != null) {
      opBuilder.setAttr("container", options.container);
    }
    if (options.sharedName != null) {
      opBuilder.setAttr("sharedName", options.sharedName);
    }
    if (options.compressionType != null) {
      opBuilder.setAttr("compressionType", options.compressionType);
    }
    return new TFRecordReader(opBuilder.build());
  }
  
  /**
   * @param container
   **/
  public static Options container(String container) {
    return new Options().container(container);
  }
  
  /**
   * @param sharedName
   **/
  public static Options sharedName(String sharedName) {
    return new Options().sharedName(sharedName);
  }
  
  /**
   * @param compressionType
   **/
  public static Options compressionType(String compressionType) {
    return new Options().compressionType(compressionType);
  }
  
  public Output<String> readerHandle() {
    return readerHandle;
  }
  
  @Override
  public Output<String> asOutput() {
    return readerHandle;
  }
  
  private Output<String> readerHandle;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private TFRecordReader(Operation operation) {
    super(operation);
    int outputIdx = 0;
    readerHandle = operation.output(outputIdx++);
  }
}
