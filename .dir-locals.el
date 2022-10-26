((nil
  (eval .(progn
          ;; (make-local-variable 'process-environment)
          ;; (setq process-environment (copy-sequence process-environment))
          ;; (setenv "PYTHONPATH" (concat (getenv "PYTHONPATH") ":/home/pupil/Documents/projects/envs/pytorch/lib/python3.8/site-packages"))
          (setq python-shell-interpreter "~/Documents/projects/envs/pytorch/bin/python")
          (setq flycheck-python-pylint-executable "~/Documents/projects/envs/pytorch/bin/python")
          (setenv "PYTHONPATH" (concat (getenv "PYTHONPATH") ":/home/pupil/Documents/upgrad/msc"))
          )
        )
 ))
