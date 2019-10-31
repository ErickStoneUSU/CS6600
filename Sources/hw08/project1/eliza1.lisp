;;; -*- Mode: Lisp; Syntax: Common-Lisp; -*-
#|
=========================================================
Module: eliza1.lisp: 
Description: Basic version of the Eliza program
bugs to vladimir dot kulyukin at usu dot edu.
=========================================================
|#

;;; (load "auxfuns.lisp")

;;; ==============================

(defun rule-pattern (rule) (first rule))
(defun rule-responses (rule) (rest rule))

;;; ==============================

#|
(defparameter *eliza-rules*
 '((((?* ?x) hello (?* ?y))      
    (How do you do.  Please state your problem.))
   (((?* ?x) I want (?* ?y))     
    (What would it mean if you got ?y)
    (Why do you want ?y) (Suppose you got ?y soon))
   (((?* ?x) if (?* ?y)) 
    (Do you really think its likely that ?y) (Do you wish that ?y)
    (What do you think about ?y) (Really-- if ?y))
   (((?* ?x) no (?* ?y))
    (You are being a bit negative) (Why not?) 
    (Are you saying "NO" just to be negative?))
   (((?* ?x) I was (?* ?y))       
    (Were you really?) (Perhaps I already knew you were ?y)
    (Why do you tell me you were ?y now?))
   (((?* ?x) I feel (?* ?y))     
    (Do you often feel ?y ?))
   (((?* ?x) I felt (?* ?y))     
    (What other feelings do you have?))))
|#

;;; ==============================

(defun eliza1 ()
  "Respond to user input using pattern matching rules."
  (loop
    (print 'eliza>)
    (write (flatten (use-eliza-rules (read))) :pretty t)))

(defun use-eliza-rules (input)
  "Find some rule with which to transform the input."
  (some #'(lambda (rule)
            (let ((result (pat-match (rule-pattern rule) input)))
              (if (not (eq result fail))
                  (sublis (switch-viewpoint result)
                          (random-elt (rule-responses rule))))))
        *eliza-rules*))

(defun switch-viewpoint (words)
  "Change I to you and vice versa, and so on."
  (sublis '((I . you) (you . I) (me . you) (am . are))
          words))

;;; ==============================

;;;> (eliza)
;;;ELIZA> (hello there)
;;;(HOW DO YOU DO. PLEASE STATE YOUR PROBLEM.) 
;;;ELIZA> (i want to test this program)
;;;(WHAT WOULD IT MEAN IF YOU GOT TO TEST THIS PROGRAM) 
;;;ELIZA> (i could see if it works)
;;;(DO YOU REALLY THINK ITS LIKELY THAT IT WORKS)
;;;ELIZA> (no not really)
;;;(ARE YOU SAYING "NO" JUST TO BE NEGATIVE?) 
;;;ELIZA> (no)
;;;(ARE YOU SAYING "NO" JUST TO BE NEGATIVE?) 
;;;ELIZA> (forget it-- i was wondering how general the program is)
;;;(WHY DO YOU TELL ME YOU WERE WONDERING HOW GENERAL THE PROGRAM IS NOW?)
;;;ELIZA> (i felt like it)
;;;(WHAT OTHER FEELINGS DO YOU HAVE?) 
;;;ELIZA> (i feel this is enough)
;;;(DO YOU OFTEN FEEL THIS IS ENOUGH ?) 
;;;ELIZA> [Abort]

;;; ==============================

