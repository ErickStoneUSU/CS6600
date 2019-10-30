;;; -*- Mode: Lisp; Syntax: Common-Lisp; -*-

#|
=========================================================
Module: load-eliza.lisp: 
Description: a load for three eliza modules.
bugs to vladimir dot kulyukin at usu dot edu.
=========================================================
|#

;;; change this parameter as needed.
(defparameter *eliza-path* "C:/Users/erick/OneDrive/Desktop/CS6600/Sources/hw08/project1/")

(defparameter *eliza-files* '("auxfuns.lisp" "eliza1.lisp" "eliza.lisp")) 

(defun load-eliza (path files)
  (mapc #'(lambda (file)
	    (load (concatenate 'string path file)))
	files))

	
