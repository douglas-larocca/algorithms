import pkg_resources
from distutils.core import setup, Extension
from distutils.command.build import build
from distutils.command.sdist import sdist
from Cython.Distutils import build_ext as _build_ext

class build_ext(_build_ext):
    def build_extensions(self):
        numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')

        for ext in self.extensions:
            if hasattr(ext, 'include_dirs') and not numpy_incl in ext.include_dirs:
                ext.include_dirs.append(numpy_incl)
        _build_ext.build_extensions(self)

setup(
    name='algorithms',
    version='0.0.1',
    description="algorithms",
    long_description="",
    classifiers=[
        "License :: OSI Approved :: GPLv3 License",
        'Programming Language :: Python :: 3.3',
    ],
    author='Douglas La Rocca',
    packages=['algorithms',
               'algorithms.numerical',
               'algorithms.combinatorial',
               'algorithms.messaging',
               'algorithms.sorting',
               'algorithms.crypto',
               'algorithms.strings',
               'algorithms.data_structures',
               'algorithms.pattern_matching',
               'algorithms.utils',
               'algorithms.geometry',
               'algorithms.searching',
               'algorithms.graphs',
               'algorithms.mathematics',
               'algorithms.sets',
              ],
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("algorithms.lib", ["algorithms/lib.pyx"]),
                 Extension("algorithms.gray_code",["algorithms/combinatorial/gray_code.pyx"]),
                 Extension("algorithms.gcc_builtin",["algorithms/utils/gcc_builtin.pyx"])]
)