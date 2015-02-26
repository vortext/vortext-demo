(ns vortext.routes.topology
  (:require [compojure.core :refer :all]
            [clojure.core.async :as async :refer [go <! >!]]
            [vortext.topologies :as topologies]
            [org.httpkit.server :as http]))

(defn topology-handler
  [name request]
  (let [c (topologies/process name request)]
    (http/with-channel request channel
      (go
        (let [result (<! c)]
          (http/send! channel (:sink result))))
      (http/on-close channel (fn [_] (async/close! c))))))

(defroutes topology-routes
  (POST "/topologies/:name" [name :as request] (topology-handler name request)))
