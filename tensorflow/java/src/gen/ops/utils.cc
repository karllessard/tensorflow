/*
 * type_evaluator.cpp
 *
 *  Created on: Mar 6, 2017
 *      Author: karl
 */
#include <cstring>

#include "tensorflow/core/platform/logging.h"
#include "types.h"

using namespace ::google::protobuf;

namespace tensorflow {

string FromUnderscoreToOtherCase(const std::string& str, bool startWithCap) {
  std::string result;
  const char joiner = '_';
  std::size_t i = 0;
  bool cap = startWithCap;
  while (i < str.size()) {
    const char c = str[i++];
    if (c == joiner) {
      cap = true;
    } else if (cap) {
      result += toupper(c);
      cap = false;
    } else {
      result += c;
    }
  }
  return result;
}

const char& EscapeDocCharacter(const char& c, string& javadoc, bool& newline, const string& newline_start) {
    if (c == '\n') {
      if (newline) {
        javadoc.append("<p>");
      }
      javadoc.push_back(c);
      javadoc.append(newline_start);
      newline = true;

    } else {
      switch (c) {
        case '<':
          javadoc.append("&lt;");
          break;
        case '>':
          javadoc.append("&gt;");
          break;
        case '&':
          javadoc.append("&amp;");
          break;
        case '"':
          javadoc.append("&quot;");
          break;
        case '\'':
          javadoc.append("&apos;");
          break;
        case '/':
          // avoids to close the Java documentation on a '*/' sequence in the documentation
          javadoc.append("&#47;");
          break;
        case '`':
          // skipped
          // FIXME do we want to enclose the quoted word with a tag instead? (e.g. <tt>)
          // If so, be aware of "triple-quotes" in some doc written using the python syntax
          break;
        default:
          javadoc.push_back(c);
          break;
      }
      newline = false;
    }
    return c;
}

string EscapeDoc(const string& doc, int indent) {
  string javadoc;
  javadoc.reserve(doc.length());
  string newline_start(indent, ' ');
  newline_start.append(" * ");
  bool newline = true;
  bool preformat = false;
  int preformat_count = 0;

  for (std::string::const_iterator it = doc.cbegin(); it != doc.cend(); ++it) {
    const char& c = EscapeDocCharacter(*it, javadoc, newline, newline_start);

    if (c == '`' || c == '"' || c == '\'') {
      if (++preformat_count >= 3) { // Handles python-style comment blocks (e.g. ```)
        if (preformat) {
          javadoc.append("</pre>");
          preformat = false;
        } else {
          javadoc.append("<pre>");
          preformat = true;
        }
      }
    } else {
      preformat_count = 0;
    }
  }
  if (!javadoc.empty() && javadoc.at(javadoc.length() - 1) != '.') {
    javadoc.push_back('.');
  }
  return javadoc;
}

} /* namespace tensorflow */
