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
package org.tensorflow.op.parsing;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class DecodeRaw<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param littleEndian
     **/
    public Options littleEndian(Boolean littleEndian) {
      this.littleEndian = littleEndian;
      return this;
    }
    
    private Boolean littleEndian;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new DecodeRaw operation to the graph.
   * 
   * @param scope Current graph scope
   * @param bytes
   * @param out_type
   * @return a new instance of DecodeRaw
   **/
  public static <T> DecodeRaw<T> create(Scope scope, Operand<String> bytes, Class<T> out_type) {
    OperationBuilder opBuilder = scope.graph().opBuilder("DecodeRaw", scope.makeOpName("DecodeRaw"));
    opBuilder.addInput(bytes.asOutput());
    opBuilder.setAttr("out_type", DataType.fromClass(out_type));
    return new DecodeRaw<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new DecodeRaw operation to the graph.
   * 
   * @param scope Current graph scope
   * @param bytes
   * @param out_type
   * @param options an object holding optional attributes values
   * @return a new instance of DecodeRaw
   **/
  public static <T> DecodeRaw<T> create(Scope scope, Operand<String> bytes, Class<T> out_type, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("DecodeRaw", scope.makeOpName("DecodeRaw"));
    opBuilder.addInput(bytes.asOutput());
    opBuilder.setAttr("out_type", DataType.fromClass(out_type));
    if (options.littleEndian != null) {
      opBuilder.setAttr("littleEndian", options.littleEndian);
    }
    return new DecodeRaw<T>(opBuilder.build());
  }
  
  /**
   * @param littleEndian
   **/
  public static Options littleEndian(Boolean littleEndian) {
    return new Options().littleEndian(littleEndian);
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
  private DecodeRaw(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
