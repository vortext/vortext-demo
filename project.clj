(defproject vortext "0.1.0-SNAPSHOT"
  :description "A vortext instance for Evidence Based Medicine"
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
  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:deprecation"]
  :dependencies [[org.clojure/clojure "1.6.0"]
                 [org.clojure/core.async "0.1.303.0-886421-alpha"]
                 [org.clojure/tools.cli "0.3.1"]

                 [log4j "1.2.17" :exclusions [javax.mail/mail
                                              javax.jms/jms
                                              com.sun.jdmk/jmxtools
                                              com.sun.jmx/jmxri]]
                 [com.taoensso/timbre "3.3.1"]

                 [environ "1.0.0"]

                 [lib-noir "0.9.5"]
                 [noir-exception "0.2.3"]
                 [selmer "0.8.0"]

                 [http-kit "2.1.19"]
                 [compojure "1.3.1"]
                 [ring/ring-devel "1.3.2"]

                 [prismatic/plumbing "0.3.7"]

                 ;; serialization libraries
                 [cheshire "5.4.0"]

                 ;; ZeroMQ

                 [org.zeromq/cljzmq "0.1.4" :exclusions [org.zeromq/jzmq]]]
  :profiles {:dev {:dependencies [[org.zeromq/jeromq "0.3.4"]]}
             :production {:dependencies [[org.zeromq/jzmq "3.1.0"]]
                          :jvm-opts ["-server"
                                     "-XX:+AggressiveOpts"
                                     "-XX:+UseG1GC"
                                     "-Djava.library.path=/usr/lib:/usr/local/lib"]
                          :env {:dev false}}
             :uberjar {:aot :all}})
