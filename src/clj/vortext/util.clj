(ns vortext.util
  (:require [clojure.string :as string]
            [clojure.walk :as walk]))


;; from https://github.com/jeremyheiler/wharf
;; A Clojure library for transforming map keys.
(defn transform-keys
  "Recursively transforms all map keys in coll with t."
  [t coll]
  (let [f (fn [[k v]] [(t k) v])]
    (walk/postwalk (fn [x] (if (map? x) (into {} (map f x)) x)) coll)))

(defn ^String capitalize
  "Converts the first character of s to upper-case. This differs from
   clojure.string/captialize because it doesn't touch the rest of s."
  [s]
  (str (.toUpperCase (subs s 0 1)) (subs s 1)))

(defn ^String uncapitalize
  "Converts the first character of s to lower-case."
  [s]
  (str (.toLowerCase (subs s 0 1)) (subs s 1)))

(defn parse-port [port]
  (when port
    (cond
      (string? port) (Integer/parseInt port)
      (number? port) port
      :else          (throw (Exception. (str "invalid port value: " port))))))
