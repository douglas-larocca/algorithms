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
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("lib", ["lib.pyx"]),
                 Extension("gray_code",["combinatorial/gray_code.pyx"]),
                 Extension("gcc_builtin",["utils/gcc_builtin.pyx"])]
)