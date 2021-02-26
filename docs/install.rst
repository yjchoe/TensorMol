============
Installation
============

.. highlight:: bash

::

$ git clone https://github.com/jparkhill/TensorMol.git
$ cd TensorMol
$ pip install --user -e .

Requirements
============
* TensorFlow_ 1.1 or newer
* Python_ 2.7 or newer
* NumPy_ 1.10 or newer (base N-dimensional array package)
* SciPy_ 0.16 or newer (library for scientific computing)
* Useful Pre-Requisites: CUDA7.5, PySCF
* To Train Minimally: ~100GB Disk 20GB memory
* To Train Realistically: 1TB Disk, GTX1070++
* To Evaluate: Normal CPU and 10GB Mem

Optional:

* ASE_
* Matplotlib_ 2.0.0 or newer (plotting)
* :mod:`tkinter` (for :mod:`ase.gui`)
* Flask_ (for :mod:`ase.db` web-interface)

.. _ASE: https://wiki.fysik.dtu.dk/ase/index.html
.. _TensorFlow: https://www.tensorflow.org/
.. _Python: http://www.python.org/
.. _NumPy: http://docs.scipy.org/doc/numpy/reference/
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _Matplotlib: http://matplotlib.org/
.. _Flask: http://flask.pocoo.org/
.. _PIP: https://pip.pypa.io/en/stable/

