/* -*- tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2; js-indent-level: 2; -*- */
define(function (require) {
  'use strict';

  var Backbone = require("backbone");
  var React = require("react");
  var _ = require("underscore");
  var FileUtil = require("spa/helpers/fileUtil");


  // Set CSRF
  var _sync = Backbone.sync;
  Backbone.sync = function(method, model, options){
    options.beforeSend = function(xhr){
      xhr.setRequestHeader('X-CSRF-Token', CSRF_TOKEN);
    };
    return _sync(method, model, options);
  };

  // Models
  var documentModel = new (require("spa/models/document"))();
  var marginaliaModel = new (require("spa/models/marginalia"))();

  // Components
  var TopBar = React.createFactory(require("jsx!components/topBar"));
  var Document = React.createFactory(require("jsx!spa/components/document"));
  var Marginalia = React.createFactory(require("jsx!spa/components/marginalia"));

  var process = function(data) {
    var upload = FileUtil.upload("/topologies/ebm", data);
    documentModel.loadFromData(data);
    upload.then(function(result) {
      var marginalia = JSON.parse(result);
      marginaliaModel.reset(marginaliaModel.parse(marginalia));
    });
  };

  var topBarComponent = React.render(
    TopBar({
      callback: process,
      accept: ".pdf",
      mimeType: /application\/(x-)?pdf|text\/pdf/
    }),
    document.getElementById("top-bar")
  );

  var documentComponent = React.render(
    Document({pdf: documentModel}),
    document.getElementById("viewer")
  );

  var marginaliaComponent = React.render(
    Marginalia({marginalia: marginaliaModel}),
    document.getElementById("marginalia")
  );

  // Dispatch logic
  // Listen to model change callbacks -> trigger updates to components
  marginaliaModel.on("all", function(e, obj) {
    switch(e) {
    case "annotations:select":
      var fingerprint = documentModel.get("fingerprint");
      documentComponent.setState({select: obj});
      break;
    case "annotations:change":
      break;
    case "annotations:add":
    case "annotations:remove":
    case "change:description":
    default:
      documentModel.setActiveAnnotations(marginaliaModel);
      marginaliaComponent.forceUpdate();
    }
  });

  documentModel.on("all", function(e, obj) {
    switch(e) {
    case "change:raw":
      var fingerprint = obj.changed.raw.pdfInfo.fingerprint;
      documentComponent.setState({
        fingerprint: fingerprint
      });
      break;
    case "change:binary":
      marginaliaModel.reset();
      break;
    case "annotation:add":
      var model = marginaliaModel.findWhere({active: true}).get("annotations");
      model.add(obj);
      break;
    case "pages:change:state":
      if(obj.get("state") > window.RenderingStates.HAS_PAGE) {
        documentModel.setActiveAnnotations(marginaliaModel);
      }
      documentComponent.forceUpdate();
      break;
    case "pages:change:annotations":
      var annotations = marginaliaModel.pluck("annotations");
      var highlighted = _.find(annotations, function(annotation) { return annotation.findWhere({highlighted: true});});
      documentComponent.setProps({highlighted: highlighted && highlighted.findWhere({highlighted: true})});
      break;
    default:
      break;
    }
  });

});
