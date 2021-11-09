(defvar m (make-array '(4 3)))
(setf *random-state* (make-random-state t))
(dotimes (i 4)
    (dotimes (j 3)
        (setf (aref m i j) (/ (- (random 10) (random 20)) (+ 1 (random 7))))))
(print m)


(defun ind-column-with-highest-sum (matrix)
    (let ((max-sum 0) (sum 0) (idx 0)) 
        (dotimes (j (second (array-dimensions matrix)))
            (dotimes (i (first (array-dimensions matrix)))
                (incf sum (aref matrix i j)))
            (when (or (= j 0) (> sum max-sum))
                (setf max-sum sum idx j))
                (setf sum 0))idx))

(print (ind-column-with-highest-sum #2A((-2 -3/7 -17/6) (-15 -5/4 -12/5))))
(print (ind-column-with-highest-sum #2A((-8 -3/2 -3/2) (-7/4 -6/5 -2/3) (-2/5 0 -13/4) (-7/4 -13 -9/2))))
(print (ind-column-with-highest-sum #2A((6 1) (1/3 -3/4) (-4 1/4))))
(print (ind-column-with-highest-sum #2A((-5/6 2 5/7 8))))
(print (ind-column-with-highest-sum #2A((-5 -9/2) (-1 -7/2) (-12 -1/3) (-6 -2/3))))


#2A((-2 -3/7 -17/6) (-15 -5/4 -12/5)) 
#2A((-8 -3/2 -3/2) (-7/4 -6/5 -2/3) (-2/5 0 -13/4) (-7/4 -13 -9/2))
#2A((6 1) (1/3 -3/4) (-4 1/4))
#2A((-5/6 2 5/7 8)) 
#2A((-5 -9/2) (-1 -7/2) (-12 -1/3) (-6 -2/3)) 
