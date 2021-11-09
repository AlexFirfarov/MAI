(defun whitespace-char-p (char)
  (member char '(#\Space #\Tab #\Newline)))

(defun russian-p (char)
  (position char "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"))

(defun word-list (string)
  (loop with len = (length string)
        for left = 0 then (1+ right)
        for right = (or (position-if #'whitespace-char-p string
                                     :start left)
                        len)
        unless (= right left)
          collect (string-right-trim ",.;:?!" (subseq string left right))
        while (< right len)))

(defun count-words-with-two-chars (char-bag sentence)
    (let ((cnt 0) (res '())) 
        (loop for word in (word-list sentence) do
            (loop for ch across word do
                (loop for ch-from-bag in char-bag do
                    (when (or   
                            (and (russian-p ch) (russian-p ch-from-bag)
                                 (or (= (russian-p ch) (russian-p ch-from-bag))
                                     (= 33 (abs (- (russian-p ch) (russian-p ch-from-bag))))))
                            (and (not (russian-p ch)) (not (russian-p ch-from-bag)) 
                                 (equalp ch ch-from-bag)))
                      (incf cnt 1))))
            (when (>= cnt 2)
                (setq res (append res (list word))))
            (setq cnt 0))res))

(print (count-words-with-two-chars '(#\А #\Д) "Сказать по правде,  я не люблю весны."))
(print (count-words-with-two-chars '(#\а #\Б #\е) "Собираясь в магазин, наденьте удобную одежду!"))
(print (count-words-with-two-chars '(#\О) "Это могло быть стихотворением"))
(print (count-words-with-two-chars '(#\w #\O #\A) "What are you   doing   now?"))
(print (count-words-with-two-chars '(#\E #\O #\r) "She is reading a book to the children."))

(print (count-words-with-two-chars '(#\O #\w #\О) "Это могло быть wot стихотворением"))