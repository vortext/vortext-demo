(ns topologies.ebm ;; This MUST be in the form topologies.<name> and cannot contain special characters.
  (:require [vortext.services :refer [js py]]
            [plumbing.core :refer :all]
            [cheshire.core :as json]))

;; The topology MUST be defined and MUST be compilable by prismatic graph.
;; The input to the topology is the Ring request, by convention called source.
;; We only return the sink to the client.
;; Make sure sink returns a valid Ring response
;; You MAY define custom serialization / deserialization, as none is done by default.
;; See the [Ring Spec](https://github.com/ring-clojure/ring/blob/master/SPEC)

(def topology
  {:source        (fnk [body] (.bytes body))
   :text           (fnk [source] (js "ebm/document_parser.js" source :timeout 4000))
   :risk-of-bias  (fnk [text] (py "ebm.risk_of_bias" text :timeout 5000))
   :sink          (fnk [risk-of-bias] (String. risk-of-bias))
   })
