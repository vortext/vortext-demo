(defproject vortext "0.1.0-SNAPSHOT"
  :description "A visualizer for predictions"
  :license {:name "GNU General Public License (GPL) v3"
            :url "https://www.gnu.org/copyleft/gpl.html"}
  :url "https://github.com/joelkuiper/vortext"
  :main vortext.core
  :source-paths ["src/clj" "resource/topologies"]
  :java-source-paths ["src/java" "resources/topologies"]
  :plugins [[lein-environ "1.0.0"]]
  :env {:broker-socket "tcp://127.0.0.1:6667"
        :default-timeout 2500,
        :heartbeat-interval 2500,
        :reconnect-timeout 2500,

        :port 8888
        :dev true}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.clojure/core.async "0.2.374"]
                 [org.clojure/tools.cli "0.3.3"]

                 [log4j "1.2.17" :exclusions [javax.mail/mail
                                              javax.jms/jms
                                              com.sun.jdmk/jmxtools
                                              com.sun.jmx/jmxri]]
                 [com.taoensso/timbre "4.3.1"]

                 [environ "1.0.2"]

                 [lib-noir "0.9.9"]
                 [noir-exception "0.2.5"]
                 [selmer "1.0.3"]

                 [http-kit "2.1.19"]
                 [compojure "1.5.0"]
                 [ring/ring-devel "1.4.0"]

                 [prismatic/plumbing "0.5.3"]

                 ;; serialization libraries
                 [cheshire "5.5.0"]

                 ;; ZeroMQ
                 [org.zeromq/cljzmq "0.1.4" :exclusions [org.zeromq/jzmq]]
                 [org.zeromq/jeromq "0.3.5"]]

  :uberjar-name "vortext-demo.jar"
  :profiles {:production {:env {:dev false}}})
