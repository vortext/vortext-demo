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
    var upload = FileUtil.upload("/topologies/gen2phen", data);
    documentModel.loadFromData(data);
    upload.then(function(result) {
      var marginalia = JSON.parse(result);
      marginaliaModel.reset(marginaliaModel.parse(marginalia));
    });
  };

  var topBarComponent = React.render(
    new TopBar({
      callback: process,
      accept: ".pdf",
      mimeType: /application\/(x-)?pdf|text\/pdf/
    }),
    document.getElementById("top-bar")
  );

  var documentComponent = React.render(
    new Document({pdf: documentModel, marginalia: marginaliaModel}),
    document.getElementById("viewer")
  );

  var marginaliaComponent = React.render(
    new Marginalia({marginalia: marginaliaModel}),
    document.getElementById("marginalia")
  );

  // Dispatch logic
  // Listen to model change callbacks -> trigger updates to components
  marginaliaModel.on("all", function(e, obj) {
    switch(e) {
    case "reset":
      documentModel.annotate(marginaliaModel.getActive());
      marginaliaComponent.forceUpdate();
      break;
    case "annotations:change":
      break;
    case "change:active":
    case "annotations:add":
    case "annotations:remove":
      documentModel.annotate(marginaliaModel.getActive());
      marginaliaComponent.forceUpdate();
      break;
    case "annotations:select":
      documentComponent.setState({select: obj});
      break;
    default:
      marginaliaComponent.forceUpdate();
    }
  });

  documentModel.on("all", function(e, obj) {
    switch(e) {
    case "change:raw":
      documentComponent.setState({
        fingerprint: documentModel.get("fingerprint")
      });
      break;
    case "change:binary":
      marginaliaModel.reset();
      break;
    case "pages:change:state":
      if(obj.get("state") == window.RenderingStates.HAS_CONTENT) {
        documentModel.annotate(marginaliaModel.getActive());
      }
      documentComponent.forceUpdate();
      break;
    case "pages:ready":
    case "pages:change:annotations":
      documentModel.annotate(marginaliaModel.getActive());
      documentComponent.forceUpdate();
      break;
    default:
      break;
    }
  });
});
