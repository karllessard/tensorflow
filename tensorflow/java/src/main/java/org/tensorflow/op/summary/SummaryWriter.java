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
package org.tensorflow.op.summary;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class SummaryWriter extends PrimitiveOp implements Operand<Object> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param sharedName
     **/
    public Options sharedName(String sharedName) {
      this.sharedName = sharedName;
      return this;
    }
    
    /**
     * @param container
     **/
    public Options container(String container) {
      this.container = container;
      return this;
    }
    
    private String sharedName;
    private String container;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new SummaryWriter operation to the graph.
   * 
   * @param scope Current graph scope
   * @return a new instance of SummaryWriter
   **/
  public static SummaryWriter create(Scope scope) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SummaryWriter", scope.makeOpName("SummaryWriter"));
    return new SummaryWriter(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new SummaryWriter operation to the graph.
   * 
   * @param scope Current graph scope
   * @param options an object holding optional attributes values
   * @return a new instance of SummaryWriter
   **/
  public static SummaryWriter create(Scope scope, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SummaryWriter", scope.makeOpName("SummaryWriter"));
    if (options.sharedName != null) {
      opBuilder.setAttr("sharedName", options.sharedName);
    }
    if (options.container != null) {
      opBuilder.setAttr("container", options.container);
    }
    return new SummaryWriter(opBuilder.build());
  }
  
  /**
   * @param sharedName
   **/
  public static Options sharedName(String sharedName) {
    return new Options().sharedName(sharedName);
  }
  
  /**
   * @param container
   **/
  public static Options container(String container) {
    return new Options().container(container);
  }
  
  public Output<?> writer() {
    return writer;
  }
  
  @Override
  @SuppressWarnings("unchecked")
  public Output<Object> asOutput() {
    return (Output<Object>) writer;
  }
  
  private Output<?> writer;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private SummaryWriter(Operation operation) {
    super(operation);
    int outputIdx = 0;
    writer = operation.output(outputIdx++);
  }
}
