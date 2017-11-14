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

public final class FixedLengthRecordReaderV2 extends PrimitiveOp implements Operand<Object> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param headerBytes
     **/
    public Options headerBytes(Integer headerBytes) {
      this.headerBytes = headerBytes;
      return this;
    }
    
    /**
     * @param footerBytes
     **/
    public Options footerBytes(Integer footerBytes) {
      this.footerBytes = footerBytes;
      return this;
    }
    
    /**
     * @param hopBytes
     **/
    public Options hopBytes(Integer hopBytes) {
      this.hopBytes = hopBytes;
      return this;
    }
    
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
     * @param encoding
     **/
    public Options encoding(String encoding) {
      this.encoding = encoding;
      return this;
    }
    
    private Integer headerBytes;
    private Integer footerBytes;
    private Integer hopBytes;
    private String container;
    private String sharedName;
    private String encoding;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new FixedLengthRecordReaderV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param recordBytes
   * @return a new instance of FixedLengthRecordReaderV2
   **/
  public static FixedLengthRecordReaderV2 create(Scope scope, Integer recordBytes) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FixedLengthRecordReaderV2", scope.makeOpName("FixedLengthRecordReaderV2"));
    opBuilder.setAttr("recordBytes", recordBytes);
    return new FixedLengthRecordReaderV2(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new FixedLengthRecordReaderV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param recordBytes
   * @param options an object holding optional attributes values
   * @return a new instance of FixedLengthRecordReaderV2
   **/
  public static FixedLengthRecordReaderV2 create(Scope scope, Integer recordBytes, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FixedLengthRecordReaderV2", scope.makeOpName("FixedLengthRecordReaderV2"));
    opBuilder.setAttr("recordBytes", recordBytes);
    if (options.headerBytes != null) {
      opBuilder.setAttr("headerBytes", options.headerBytes);
    }
    if (options.footerBytes != null) {
      opBuilder.setAttr("footerBytes", options.footerBytes);
    }
    if (options.hopBytes != null) {
      opBuilder.setAttr("hopBytes", options.hopBytes);
    }
    if (options.container != null) {
      opBuilder.setAttr("container", options.container);
    }
    if (options.sharedName != null) {
      opBuilder.setAttr("sharedName", options.sharedName);
    }
    if (options.encoding != null) {
      opBuilder.setAttr("encoding", options.encoding);
    }
    return new FixedLengthRecordReaderV2(opBuilder.build());
  }
  
  /**
   * @param headerBytes
   **/
  public static Options headerBytes(Integer headerBytes) {
    return new Options().headerBytes(headerBytes);
  }
  
  /**
   * @param footerBytes
   **/
  public static Options footerBytes(Integer footerBytes) {
    return new Options().footerBytes(footerBytes);
  }
  
  /**
   * @param hopBytes
   **/
  public static Options hopBytes(Integer hopBytes) {
    return new Options().hopBytes(hopBytes);
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
   * @param encoding
   **/
  public static Options encoding(String encoding) {
    return new Options().encoding(encoding);
  }
  
  public Output<?> readerHandle() {
    return readerHandle;
  }
  
  @Override
  @SuppressWarnings("unchecked")
  public Output<Object> asOutput() {
    return (Output<Object>) readerHandle;
  }
  
  private Output<?> readerHandle;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private FixedLengthRecordReaderV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    readerHandle = operation.output(outputIdx++);
  }
}
