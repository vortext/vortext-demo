(ns vortext.routes.topology
  (:require [compojure.core :refer :all]
            [clojure.core.async :as async :refer [go <! >!]]
            [vortext.topologies :as topologies]
            [org.httpkit.server :as http]))

(defn topology-handler
  [name request]
  (let [c (topologies/process name request)]
    (http/with-channel request channel
      (async/go
        (let [resp (:sink (<! c))]
          (http/send! channel resp))))))

(defroutes topology-routes
  (POST "/topologies/:name" [name :as request] (topology-handler name request)))
