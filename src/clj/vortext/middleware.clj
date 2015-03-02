(ns vortext.middleware
  (:require [taoensso.timbre :as timbre]
            [environ.core :refer [env]]
            [noir.util.middleware :refer :all]
            [selmer.middleware :refer [wrap-error-page]]
            [ring.middleware.stacktrace :refer :all]
            [noir-exception.core :refer [wrap-internal-error wrap-exceptions]]))

(def common-middleware
  [wrap-strip-trailing-slash
   wrap-stacktrace])

(def development-middleware
  [wrap-error-page
   wrap-exceptions])

(def production-middleware
  [#(wrap-internal-error % :log (fn [e] (timbre/error e)))])

(defn load-middleware []
  (concat common-middleware
          (if (env :dev)
            development-middleware
            production-middleware)))
