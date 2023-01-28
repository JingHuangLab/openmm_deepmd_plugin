from distutils.core import setup
from distutils.extension import Extension
import os
import platform

openmm_dir = '/opt/conda'
deepmd_dir = '/usr/local/libdeepmd/libdeepmd_c'
DeepmdPlugin_header_dir = '/mnt/openmmapi/include'
DeepmdPlugin_library_dir = '/mnt/build4docker_test'

os.environ["CC"] = "/opt/conda/bin/x86_64-conda-linux-gnu-cc"
os.environ["CXX"] = "/opt/conda/bin/x86_64-conda-linux-gnu-c++"

extra_compile_args = []
extra_link_args = []


# setup extra compile and link arguments on Mac
if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='OpenMMDeepmdPlugin._OpenMMDeepmdPlugin',
                      sources=['OpenMMDeepmdPluginWrapper.cpp'],
                      libraries=['OpenMM', 'OpenMMDeepmd'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), os.path.join(deepmd_dir, 'include'), DeepmdPlugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), os.path.join(deepmd_dir, 'lib'), DeepmdPlugin_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )



setup(name='OpenMMDeepmdPlugin',
      version="c88f0a2",
      ext_modules=[extension],
      packages=['OpenMMDeepmdPlugin', "OpenMMDeepmdPlugin.tests"],
      package_data={"OpenMMDeepmdPlugin":['data/*.pb', 'data/*.pdb']},
     )
