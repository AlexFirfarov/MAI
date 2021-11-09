(defun rec-1 (x)
    (if x (cons (- (car x) 1) (rec-1 (cdr x))) nil))

(defun map-1 (x)
    (mapcar #'1- x))
