(defclass cart ()               
 ((x :initarg :x :reader cart-x)  
  (y :initarg :y :reader cart-y)))

(defclass line ()
 ((start :initarg :start :accessor line-start)
  (end   :initarg :end   :accessor line-end)))

(defvar ln)
(setq ln (make-instance 'line
           :start (make-instance 'cart :x 7 :y 0)
           :end (make-instance 'cart :x 0 :y 7)))

(setq ln (make-instance 'line
           :start (make-instance 'cart :x 7 :y -4)
           :end (make-instance 'cart :x 5 :y 12)))

(setq ln (make-instance 'line
           :start (make-instance 'cart :x -3.2 :y 0)
           :end (make-instance 'cart :x 8.4 :y -10.1)))

(setq ln (make-instance 'line
           :start (make-instance 'cart :x 16 :y 16)
           :end (make-instance 'cart :x 0 :y 0)))

(setq ln (make-instance 'line
           :start (make-instance 'cart :x 12.5 :y 8.3)
           :end (make-instance 'cart :x 4.4 :y -12.8)))

(defun line-length (line)
    (sqrt (+ (expt (- (cart-x (line-end line)) (cart-x (line-start line))) 2)
             (expt (- (cart-y (line-end line)) (cart-y (line-start line))) 2))))

(line-length ln)