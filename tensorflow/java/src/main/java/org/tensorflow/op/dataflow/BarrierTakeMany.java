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
package org.tensorflow.op.dataflow;

import java.util.Arrays;
import org.tensorflow.DataType;
import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class BarrierTakeMany extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param allowSmallBatch
     **/
    public Options allowSmallBatch(Boolean allowSmallBatch) {
      this.allowSmallBatch = allowSmallBatch;
      return this;
    }
    
    /**
     * @param waitForIncomplete
     **/
    public Options waitForIncomplete(Boolean waitForIncomplete) {
      this.waitForIncomplete = waitForIncomplete;
      return this;
    }
    
    /**
     * @param timeoutMs
     **/
    public Options timeoutMs(Integer timeoutMs) {
      this.timeoutMs = timeoutMs;
      return this;
    }
    
    private Boolean allowSmallBatch;
    private Boolean waitForIncomplete;
    private Integer timeoutMs;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new BarrierTakeMany operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param numElements
   * @param componentTypes
   * @return a new instance of BarrierTakeMany
   **/
  public static BarrierTakeMany create(Scope scope, Operand<String> handle, Operand<Integer> numElements, List<DataType> componentTypes) {
    OperationBuilder opBuilder = scope.graph().opBuilder("BarrierTakeMany", scope.makeOpName("BarrierTakeMany"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInput(numElements.asOutput());
    opBuilder.setAttr("componentTypes", componentTypes.toArray(new DataType[componentTypes.size()]));
    return new BarrierTakeMany(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new BarrierTakeMany operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param numElements
   * @param componentTypes
   * @param options an object holding optional attributes values
   * @return a new instance of BarrierTakeMany
   **/
  public static BarrierTakeMany create(Scope scope, Operand<String> handle, Operand<Integer> numElements, List<DataType> componentTypes, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("BarrierTakeMany", scope.makeOpName("BarrierTakeMany"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInput(numElements.asOutput());
    opBuilder.setAttr("componentTypes", componentTypes.toArray(new DataType[componentTypes.size()]));
    if (options.allowSmallBatch != null) {
      opBuilder.setAttr("allowSmallBatch", options.allowSmallBatch);
    }
    if (options.waitForIncomplete != null) {
      opBuilder.setAttr("waitForIncomplete", options.waitForIncomplete);
    }
    if (options.timeoutMs != null) {
      opBuilder.setAttr("timeoutMs", options.timeoutMs);
    }
    return new BarrierTakeMany(opBuilder.build());
  }
  
  /**
   * @param allowSmallBatch
   **/
  public static Options allowSmallBatch(Boolean allowSmallBatch) {
    return new Options().allowSmallBatch(allowSmallBatch);
  }
  
  /**
   * @param waitForIncomplete
   **/
  public static Options waitForIncomplete(Boolean waitForIncomplete) {
    return new Options().waitForIncomplete(waitForIncomplete);
  }
  
  /**
   * @param timeoutMs
   **/
  public static Options timeoutMs(Integer timeoutMs) {
    return new Options().timeoutMs(timeoutMs);
  }
  
  public Output<Long> indices() {
    return indices;
  }
  
  public Output<String> keys() {
    return keys;
  }
  
  public List<Output<DataType>> values() {
    return values;
  }
  
  private Output<Long> indices;
  private Output<String> keys;
  private List<Output<DataType>> values;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private BarrierTakeMany(Operation operation) {
    super(operation);
    int outputIdx = 0;
    indices = operation.output(outputIdx++);
    keys = operation.output(outputIdx++);
    int valuesLength = operation.outputListLength("values");
    values = Arrays.asList((Output<DataType>[])operation.outputList(outputIdx, valuesLength));
    outputIdx += valuesLength;
  }
}
