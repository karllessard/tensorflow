package org.tensorflow.op.image;

import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.Scope;
import org.tensorflow.op.SingleResultNode;

public class DecodeJpeg extends SingleResultNode {

  public static class Builder extends SingleResultNode.Builder<DecodeJpeg> {

    Builder(Scope scope, Output contents) {
      super(scope, OP_TYPE);
      this.contents = contents;
    }

    public Builder withChannels(int channels) {
      this.channels = channels;
      return this;
    }

    public Builder withRatio(int ratio) {
      this.ratio = ratio;
      return this;
    }

    public Builder withFancyUpscaling(boolean fancyUpscaling) {
      this.fancyUpscaling = fancyUpscaling;
      return this;
    }

    public Builder withTryRecoverTruncated(boolean tryRecoverTruncated) {
      this.tryRecoverTruncated = tryRecoverTruncated;
      return this;
    }

    public Builder withAcceptableFraction(float acceptableFraction) {
      this.acceptableFraction = acceptableFraction;
      return this;
    }

    public Builder withDctMethods(String dctMethods) {
      this.dctMethods = dctMethods;
      return this;
    }

    @Override
    protected DecodeJpeg buildOp(OperationBuilder opBuilder) {
      return new DecodeJpeg(opBuilder
            .addInput(contents)
            .setAttr("channels", channels)
            .setAttr("ratio", ratio)
            .setAttr("fancyUpscaling", fancyUpscaling)
            .setAttr("tryRecoverTruncated", tryRecoverTruncated)
            .setAttr("acceptableFraction", acceptableFraction)
            .setAttr("dctMethods", dctMethods)
            .build());
    }

    private final Output contents;
    private int channels = 0;
    private int ratio = 1;
    private boolean fancyUpscaling = true;
    private boolean tryRecoverTruncated = false;
    private float acceptableFraction = 1.0f;
    private String dctMethods = "";
  }
  
  public Output image() {
    return op().output(0);
  }

  private DecodeJpeg(Operation op) {
    super(op);
  }

  private static final String OP_TYPE = "ExpandDims";
}
