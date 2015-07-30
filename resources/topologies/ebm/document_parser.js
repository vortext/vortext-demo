var fs = require('fs');
var atob = require('atob');
var _ = require('underscore');
var Q = require('q');

global.window = global;
global.navigator = { userAgent: 'node' };
global.PDFJS = {};
global.XMLHttpRequest = function() {}; // we don't really need this, but we could stub it with xhr2 or similar
global.DOMParser = require('./domparsermock.js').DOMParserMock;


PDFJS.workerSrc = true;

require('./pdfjs/pdf.combined.js');

function textContent(pdf, content) {
  var pages = [];

  for (var i = 0; i < content.length; i++) {
    var page = content[i];
    var items = page.items;
    var pageText = "";
    for (var j = 0; j < items.length; j++) {
      var item = items[j];
      pageText += (item.str + " ");
    }
    pages.push(pageText);
  }

  var fingerprint = pdf.pdfInfo.fingerprint;
  return {"fingerprint": fingerprint,
          "pages": pages };
}

function convertToDocument(payload) {
  return PDFJS.getDocument(payload).then(function (pdf) {
    var pages = _.map(_.range(1, pdf.numPages + 1), function(pageNr) {
      return pdf.getPage(pageNr);
    });

    return Q.all(_.invoke(pages, "then", function(page) {
      return page.getTextContent();
    })).then(function(contents) {
      return textContent(pdf, contents);
    });
  });
};

function handler(payload) {
  var pdf = new Uint8Array(Buffer(payload, "binary"));
  var document = convertToDocument(pdf);
  return document.then(function(doc) {
    return JSON.stringify(doc);
  });
}

module.exports = handler;
