(ns vortext.handler
  (:require [compojure.core :refer :all]
            [environ.core :refer [env]]
            [compojure.route :as route]
            [compojure.handler :as handler]
            [noir.util.middleware :refer [app-handler]]
            [vortext.middleware :refer :all]
            [taoensso.timbre :as timbre]
            [taoensso.timbre.appenders.rotor :as rotor]
            [ring.util.response :as response]
            [ring.middleware.anti-forgery :refer [*anti-forgery-token*]]
            [vortext.routes.topology :refer [topology-routes]]
            [vortext.routes.home :refer [home-routes]]
            [vortext.services :as services]))

(defroutes
  app-routes
  (route/resources "/static")
  (route/not-found "Page not found"))

(defn init!
  "init will be called once when app is deployed as a servlet on an
  app server such as Tomcat put any initialization code here"
  []
  (timbre/set-config!
   [:appenders :rotor]
   {:enabled? true,
    :async? false,
    :fn rotor/appender-fn})
  (timbre/set-config!
   [:shared-appender-config :rotor]
   {:path "app.log", :max-size (* 512 1024), :backlog 10})
  (if (env :dev) (selmer.parser/cache-off!))
  (selmer.parser/add-tag! :csrf-token (fn [_ _] *anti-forgery-token*))
  (services/start!)
  (timbre/info "started successfully"))

(defn destroy!
  "destroy will be called when your application
  shuts down, put any clean up code here"
  []
  (timbre/info "shutting down...")
  (services/shutdown!)
  (timbre/info "shutdown complete!"))

(def web-routes
  [home-routes
   topology-routes
   app-routes])

(def app
  (app-handler
   web-routes
   :middleware (load-middleware)))
