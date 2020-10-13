# /bin/bash
cd cocoapi/PythonAPI
python setup.py build_ext install
cd ..
cd apex
python setup.py install --cuda_ext --cpp_ext
cd ..
cd sg
python setup.py build develop
cd ..