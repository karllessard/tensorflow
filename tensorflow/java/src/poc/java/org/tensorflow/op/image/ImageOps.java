package org.tensorflow.op.image;

import org.tensorflow.InputSource;
import org.tensorflow.op.Scope;

public class ImageOps {
  
  public ImageOps(Scope scope) {
    this.scope = scope;
  }
  
  public DecodeJpeg.Builder decodeJpeg(InputSource contents) {
    return new DecodeJpeg.Builder(scope, contents.input());
  }

  private final Scope scope;
}
