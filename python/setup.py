from distutils.core import setup
from distutils.extension import Extension
import os
import platform

openmm_dir = '@OPENMM_DIR@'
deepmd_dir = '@DEEPMD_DIR@'
DeepmdPlugin_header_dir = '@DEEPMDPLUGIN_HEADER_DIR@'
DeepmdPlugin_library_dir = '@DEEPMDPLUGIN_LIBRARY_DIR@'

os.environ["CC"] = "@CMAKE_C_COMPILER@"
os.environ["CXX"] = "@CMAKE_CXX_COMPILER@"

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
      version="@GIT_HASH@",
      ext_modules=[extension],
      packages=['OpenMMDeepmdPlugin', "OpenMMDeepmdPlugin.tests"],
      package_data={"OpenMMDeepmdPlugin":['data/*.pb', 'data/*.pdb']},
     )
