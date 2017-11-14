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
import org.tensorflow.types.UInt8;

public final class DecodeGif extends PrimitiveOp implements Operand<UInt8> {
  
  /**
   * Factory method to create a class to wrap a new DecodeGif operation to the graph.
   * 
   * @param scope Current graph scope
   * @param contents
   * @return a new instance of DecodeGif
   **/
  public static DecodeGif create(Scope scope, Operand<String> contents) {
    OperationBuilder opBuilder = scope.graph().opBuilder("DecodeGif", scope.makeOpName("DecodeGif"));
    opBuilder.addInput(contents.asOutput());
    return new DecodeGif(opBuilder.build());
  }
  
  public Output<UInt8> image() {
    return image;
  }
  
  @Override
  public Output<UInt8> asOutput() {
    return image;
  }
  
  private Output<UInt8> image;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private DecodeGif(Operation operation) {
    super(operation);
    int outputIdx = 0;
    image = operation.output(outputIdx++);
  }
}
