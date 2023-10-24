from setuptools import setup
pname='mockgen'
setup(name=pname,
      version='0.1',
      description='Extragalactic mock sky generation',
      url='http://github.com/exgalsky/mockgen',
      author='exgalsky collaboration',
      license_files = ('LICENSE',),
      packages=[pname],
      entry_points ={ 
        'console_scripts': [ 
          'xgmockgen = mockgen.command_line_interface:main'
        ]
      }, 
      zip_safe=False)
