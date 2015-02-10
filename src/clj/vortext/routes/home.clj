(ns vortext.routes.home
  (:require [compojure.core :refer :all]
            [vortext.layout :as layout]))


(defroutes home-routes
  (GET "/" [] (layout/render-to-response "index.html" {})))
