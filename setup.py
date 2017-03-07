from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['hfts_grasp_planner'],
    package_dir={'': 'src'},
    requires=['rospy', 'numpy', 'yaml', 'rtree', 'tf', 'stl',
              'sklearn', 'scipy', 'igraph', 'matplotlib', 'openravepy', 'rospkg']
)

setup(**setup_args)