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

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.op.Operands;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class DecodeCSV extends PrimitiveOp implements Iterable<Operand<Object>> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param fieldDelim
     **/
    public Options fieldDelim(String fieldDelim) {
      this.fieldDelim = fieldDelim;
      return this;
    }
    
    /**
     * @param useQuoteDelim
     **/
    public Options useQuoteDelim(Boolean useQuoteDelim) {
      this.useQuoteDelim = useQuoteDelim;
      return this;
    }
    
    /**
     * @param naValue
     **/
    public Options naValue(String naValue) {
      this.naValue = naValue;
      return this;
    }
    
    private String fieldDelim;
    private Boolean useQuoteDelim;
    private String naValue;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new DecodeCSV operation to the graph.
   * 
   * @param scope Current graph scope
   * @param records
   * @param recordDefaults
   * @return a new instance of DecodeCSV
   **/
  public static DecodeCSV create(Scope scope, Operand<String> records, Iterable<Operand<?>> recordDefaults) {
    OperationBuilder opBuilder = scope.graph().opBuilder("DecodeCSV", scope.makeOpName("DecodeCSV"));
    opBuilder.addInput(records.asOutput());
    opBuilder.addInputList(Operands.asOutputs(recordDefaults));
    return new DecodeCSV(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new DecodeCSV operation to the graph.
   * 
   * @param scope Current graph scope
   * @param records
   * @param recordDefaults
   * @param options an object holding optional attributes values
   * @return a new instance of DecodeCSV
   **/
  public static DecodeCSV create(Scope scope, Operand<String> records, Iterable<Operand<?>> recordDefaults, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("DecodeCSV", scope.makeOpName("DecodeCSV"));
    opBuilder.addInput(records.asOutput());
    opBuilder.addInputList(Operands.asOutputs(recordDefaults));
    if (options.fieldDelim != null) {
      opBuilder.setAttr("fieldDelim", options.fieldDelim);
    }
    if (options.useQuoteDelim != null) {
      opBuilder.setAttr("useQuoteDelim", options.useQuoteDelim);
    }
    if (options.naValue != null) {
      opBuilder.setAttr("naValue", options.naValue);
    }
    return new DecodeCSV(opBuilder.build());
  }
  
  /**
   * @param fieldDelim
   **/
  public static Options fieldDelim(String fieldDelim) {
    return new Options().fieldDelim(fieldDelim);
  }
  
  /**
   * @param useQuoteDelim
   **/
  public static Options useQuoteDelim(Boolean useQuoteDelim) {
    return new Options().useQuoteDelim(useQuoteDelim);
  }
  
  /**
   * @param naValue
   **/
  public static Options naValue(String naValue) {
    return new Options().naValue(naValue);
  }
  
  public List<Output<?>> output() {
    return output;
  }
  
  @Override
  @SuppressWarnings({"rawtypes", "unchecked"})
  public Iterator<Operand<Object>> iterator() {
    return (Iterator) output.iterator();
  }
  
  private List<Output<?>> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private DecodeCSV(Operation operation) {
    super(operation);
    int outputIdx = 0;
    int outputLength = operation.outputListLength("output");
    output = Arrays.asList(operation.outputList(outputIdx, outputLength));
    outputIdx += outputLength;
  }
}
