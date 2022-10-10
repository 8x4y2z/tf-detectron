((nil
  (eval .(progn
          (make-local-variable 'process-environment)
          (setq process-environment (copy-sequence process-environment))
          (setenv "PYTHONPATH" "/home/pupil/Documents/upgrad/msc:$PYTHONPATH")))
 ))
